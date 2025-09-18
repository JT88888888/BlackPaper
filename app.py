import os, re, time, json, hashlib, random, requests, pandas as pd, feedparser, streamlit as st
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ----------------- Config -----------------
DEFAULT_WATCHLIST = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","SPY","QQQ","IWM","DIA","XLK","XLF","SMH","SOXX"]
DAYS_BACK = 3                      # window to pull
BATCH_SIZE = 80                    # how many tickers to query per run (control API load)
FINNHUB_PAUSE = 0.05               # small sleep between requests

# ------------- Helpers & Rules -----------
def sha(s): return hashlib.sha1(s.encode()).hexdigest()
def clamp(x,a,b): return max(a, min(b, x))

CAT_RULES = [
    ("EARNINGS",   r"\b(earnings|results|eps|revenue|profit|loss|beat|miss)\b"),
    ("GUIDANCE",   r"\b(guidance|outlook|forecast|update)\b"),
    ("M&A",        r"\b(acquire|acquisition|merger|buyout|deal|takeover)\b"),
    ("REGULATORY", r"\b(approval|fda|sec|investigation|fine|lawsuit|probe|ruling|sanction)\b"),
    ("PRODUCT",    r"\b(launch|product|platform|partnership|contract|customer|rollout)\b"),
    ("MGMT_CHANGE",r"\b(ceo|cfo|cto|resigns?|steps down|appoints?)\b"),
    ("INSIDER",    r"\b(insider|buyback|repurchase)\b"),
    ("MACRO",      r"\b(rate|inflation|tariff|sanction|geopolit|jobs|cpi|ppi|gdp)\b"),
]
def tag_catalyst(text):
    t=text.lower()
    for label,pat in CAT_RULES:
        if re.search(pat,t): return label
    return "OTHER"

def confidence_from(text, sent):
    # simple 0..1 confidence from sentiment strength + length
    strength = abs(sent)
    length_hint = clamp(len(text)/800, 0, 1)
    return round(0.6*strength + 0.4*length_hint, 2), length_hint

def score_move(sent, cat, length_hint):
    cat_w = {"GUIDANCE":18,"EARNINGS":15,"M&A":22,"REGULATORY":20,"PRODUCT":10,"MGMT_CHANGE":8,"INSIDER":7,"MACRO":6,"OTHER":4}
    base = 50 + 10*length_hint
    t1 = clamp(base + 25*sent + cat_w.get(cat,4), 0, 100)
    t5 = clamp(t1 + (5 if cat in {"GUIDANCE","REGULATORY","M&A"} else 0), 0, 100)
    return int(round(t1)), int(round(t5))

# --------------- Data sources ------------
def finnhub_company_news(ticker, token, days):
    fr = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
    to = datetime.utcnow().date().isoformat()
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={fr}&to={to}&token={token}"
    try:
        r = requests.get(url, timeout=20); r.raise_for_status()
        for x in r.json():
            ts = datetime.utcfromtimestamp(x.get("datetime",0)).isoformat()
            yield {
                "ticker": ticker,
                "source": "FINNHUB",
                "title": x.get("headline","") or "",
                "url": x.get("url","") or "",
                "published_at": ts,
                "snippet": (x.get("summary") or "")[:600]
            }
    except Exception:
        return

def edgar_company_rss(ticker, days):
    # ETFs rarely file; companies do. Still safe to call.
    feed = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&owner=exclude&count=100&output=atom"
    d = feedparser.parse(feed)
    cutoff = datetime.utcnow() - timedelta(days=days)
    for e in d.entries[:100]:
        pub = e.get("published","")
        try:
            dt = datetime.strptime(pub[:19], "%Y-%m-%dT%H:%M:%S")
        except Exception:
            continue
        if dt < cutoff: 
            continue
        yield {
            "ticker": ticker,
            "source": "EDGAR",
            "title": e.title,
            "url": e.link,
            "published_at": dt.isoformat(),
            "snippet": (e.get("summary","") or "")[:600]
        }

# --------------- Streamlit UI ------------
st.set_page_config(page_title="Stock Scout – Auto Top Picks", layout="wide")
st.title("Stock Scout – Auto-Filtered High-Potential Stocks")

# Load watchlist (tickers.csv if available)
wl_path = "tickers.csv"
if os.path.exists(wl_path):
    wl = [ln.strip().split("#")[0] for ln in open(wl_path) if ln.strip() and not ln.strip().startswith("#")]
    WATCHLIST = [t.upper() for t in wl]
else:
    WATCHLIST = DEFAULT_WATCHLIST

st.sidebar.header("Universe")
st.sidebar.write(f"Loaded **{len(WATCHLIST)}** symbols from `tickers.csv`" if os.path.exists(wl_path) else f"Using default list (**{len(WATCHLIST)}** symbols).")
days_back = st.sidebar.slider("Days back", 1, 7, DAYS_BACK)
auto_min_t1 = st.sidebar.slider("Auto Top Picks: min T+1", 60, 100, 85)
auto_min_conf = st.sidebar.slider("Auto Top Picks: min confidence", 0.0, 1.0, 0.7)

token = os.getenv("FINNHUB_TOKEN","").strip()
if not token:
    st.warning("Add your FINNHUB_TOKEN in Streamlit **Manage app → Secrets** to enable Finnhub news.")

# Batch through large lists to respect free limits
random.seed(datetime.utcnow().hour)  # rotate batch hourly
universe = WATCHLIST.copy()
random.shuffle(universe)
batch = universe[:BATCH_SIZE]

rows, seen = [], set()
for t in batch:
    if token:
        for it in finnhub_company_news(t, token, days_back):
            k = sha(it["source"]+it["title"]+it["published_at"])
            if k in seen: continue
            seen.add(k); rows.append(it)
            time.sleep(FINNHUB_PAUSE)
    for it in edgar_company_rss(t, days_back):
        k = sha(it["source"]+it["title"]+it["published_at"])
        if k in seen: continue
        seen.add(k); rows.append(it)
        time.sleep(0.02)

if not rows:
    st.info("No items yet. Increase 'Days back' or add more symbols to tickers.csv.")
    st.stop()

# NLP scoring
df = pd.DataFrame(rows)
an = SentimentIntensityAnalyzer()
texts = (df["title"].fillna("") + ". " + df["snippet"].fillna(""))

pol, conf, cat, t1s, t5s = [], [], [], [], []
for txt in texts:
    s = an.polarity_scores(txt)["compound"]
    c = tag_catalyst(txt)
    cnf, length_hint = confidence_from(txt, s)
    t1, t5 = score_move(s, c, length_hint)
    pol.append(round(s,3)); conf.append(cnf); cat.append(c); t1s.append(t1); t5s.append(t5)

df["catalyst"]=cat; df["polarity"]=pol; df["confidence"]=conf; df["T+1"]=t1s; df["T+5"]=t5s
df.sort_values(["T+1","published_at"], ascending=[False, False], inplace=True)

# ---- Auto Top Picks (AI filtered) ----
st.subheader("Auto Top Picks")
picks = df[(df["T+1"]>=auto_min_t1) & (df["confidence"]>=auto_min_conf)].copy()
st.write(f"{len(picks)} picks (T+1 ≥ {auto_min_t1}, confidence ≥ {auto_min_conf}) from a batch of {len(batch)} symbols")
st.dataframe(
    picks[["published_at","ticker","catalyst","polarity","confidence","T+1","T+5","title","source","url"]]
    .rename(columns={"url":"source_link"}).reset_index(drop=True),
    use_container_width=True
)

# ---- Full feed + filters ----
st.subheader("Full Feed (current batch)")
tickers = sorted(df["ticker"].unique())
sel_t = st.multiselect("Tickers", tickers, default=[])
min_t1 = st.slider("Min T+1 score", 0, 100, 70)
sel_cat = st.multiselect("Catalysts", sorted(df["catalyst"].unique()), default=[])

q = df.copy()
if sel_t:   q = q[q["ticker"].isin(sel_t)]
if sel_cat: q = q[q["catalyst"].isin(sel_cat)]
q = q[q["T+1"] >= min_t1]

st.dataframe(
    q[["published_at","ticker","catalyst","polarity","confidence","T+1","T+5","title","source","url"]]
    .rename(columns={"url":"source_link"}).reset_index(drop=True),
    use_container_width=True
)

st.caption("Heuristic model for research only. Batch size rotates hourly to cover a large universe within free API limits.")
