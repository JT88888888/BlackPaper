import os, re, time, json, hashlib, requests, pandas as pd, feedparser, streamlit as st
from datetime import datetime, timedelta, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -----------------------------
# CONFIG (editable in sidebar)
# -----------------------------
DEFAULT_WATCHLIST = ["AAPL","MSFT","NVDA","TSLA","AMZN"]
DAYS_BACK = 3  # pull this many days back

# -----------------------------
# Helpers
# -----------------------------
def sha(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()

def clamp(x, a, b): 
    return max(a, min(b, x))

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
def tag_catalyst(text: str) -> str:
    t = text.lower()
    for label, pat in CAT_RULES:
        if re.search(pat, t): 
            return label
    return "OTHER"

def score_move(sent: float, cat: str, length_hint: float):
    # sent in [-1,1]; length_hint ~ [0..1] from text size
    cat_w = {"GUIDANCE":18,"EARNINGS":15,"M&A":22,"REGULATORY":20,"PRODUCT":10,"MGMT_CHANGE":8,"INSIDER":7,"MACRO":6,"OTHER":4}
    base = 50 + 10*length_hint  # more specific text => slightly higher base
    t1 = clamp(base + 25*sent + cat_w.get(cat,4), 0, 100)
    t5 = clamp(t1 + (5 if cat in {"GUIDANCE","REGULATORY","M&A"} else 0), 0, 100)
    return int(round(t1)), int(round(t5))

def confidence_from(text: str, sent_compound: float):
    # simple 0..1 confidence: combine signal strength and text length
    strength = abs(sent_compound)            # 0..1
    length_hint = clamp(len(text)/800, 0, 1) # 0..1
    return round(0.6*strength + 0.4*length_hint, 2), length_hint

# -----------------------------
# Data sources
# -----------------------------
def finnhub_company_news(ticker: str, token: str, days: int):
    fr = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
    to = datetime.utcnow().date().isoformat()
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={fr}&to={to}&token={token}"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
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

def edgar_company_rss(ticker: str, days: int):
    feed = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&owner=exclude&count=100&output=atom"
    d = feedparser.parse(feed)
    # limit by date window when possible
    cutoff = datetime.utcnow() - timedelta(days=days)
    for e in d.entries[:100]:
        pub = e.get("published","")
        try:
            dt = datetime.strptime(pub[:19], "%Y-%m-%dT%H:%M:%S")
        except Exception:
            dt = datetime.utcnow()
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

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Stock Scout – Taurient Lite", layout="wide")
st.title("Stock Scout – High-Impact Signals (Taurient-lite)")

# Sidebar config
st.sidebar.header("Watchlist")
wl_txt = st.sidebar.text_area(
    "Tickers (one per line)", 
    value="\n".join(DEFAULT_WATCHLIST), height=150
)
WATCHLIST = [t.strip().upper() for t in wl_txt.splitlines() if t.strip()]
days_back = st.sidebar.slider("Days back", 1, 7, DAYS_BACK, help="How far back to fetch news/filings")

token = os.getenv("FINNHUB_TOKEN","").strip()
if not token:
    st.warning("Add your FINNHUB_TOKEN in Streamlit **Manage app → Secrets** to enable Finnhub news.")

# Pull data
rows, seen = [], set()
for t in WATCHLIST[:150]:  # safety cap
    if token:
        for it in finnhub_company_news(t, token, days_back):
            key = sha(it["source"]+it["title"]+it["published_at"])
            if key in seen: 
                continue
            seen.add(key); rows.append(it)
            time.sleep(0.05)
    # SEC EDGAR filings (always free)
    for it in edgar_company_rss(t, days_back):
        key = sha(it["source"]+it["title"]+it["published_at"])
        if key in seen: 
            continue
        seen.add(key); rows.append(it)
        time.sleep(0.05)

if not rows:
    st.info("No items yet. Try adding more tickers or wait a minute for feeds to return.")
    st.stop()

# Build dataframe + NLP
df = pd.DataFrame(rows)
an = SentimentIntensityAnalyzer()
texts = (df["title"].fillna("") + ". " + df["snippet"].fillna(""))
sent_vals = []
conf_vals = []
cats = []
t1s, t5s = [], []

for txt in texts:
    s = an.polarity_scores(txt)["compound"]  # −1..+1
    conf, length_hint = confidence_from(txt, s)  # 0..1 + length proxy
    c = tag_catalyst(txt)
    t1, t5 = score_move(s, c, length_hint)
    sent_vals.append(round(s, 3))
    conf_vals.append(conf)
    cats.append(c)
    t1s.append(t1); t5s.append(t5)

df["polarity"] = sent_vals
df["confidence"] = conf_vals
df["catalyst"] = cats
df["T+1"] = t1s
df["T+5"] = t5s

# Sort by impact (T+1 desc then recency)
df.sort_values(["T+1","published_at"], ascending=[False, False], inplace=True)

# Filters
st.subheader("Filters")
tickers = sorted(df["ticker"].unique())
sel_t = st.multiselect("Tickers", tickers, default=[])
min_t1 = st.slider("Min T+1 score", 0, 100, 70)
sel_cat = st.multiselect("Catalysts", sorted(df["catalyst"].unique()), default=[])

q = df.copy()
if sel_t:   q = q[q["ticker"].isin(sel_t)]
if sel_cat: q = q[q["catalyst"].isin(sel_cat)]
q = q[q["T+1"] >= min_t1]

# Display
cols = ["published_at","ticker","catalyst","polarity","confidence","T+1","T+5","title","source","url"]
st.dataframe(
    q[cols].rename(columns={"url":"source_link"}).reset_index(drop=True),
    use_container_width=True
)

st.caption("Polarity: −1..+1, Confidence: 0..1. Scores are heuristic and for research only.")
