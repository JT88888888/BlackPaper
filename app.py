import os, time, hashlib, requests, pandas as pd, feedparser, streamlit as st
from datetime import datetime, timedelta, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---- SETTINGS (edit these) ----
WATCHLIST = ["AAPL","MSFT","NVDA","TSLA","AMZN"]  # add/remove tickers
DAYS_BACK = 2                                     # how far back to fetch

# ---- HELPERS ----
def sha(s): return hashlib.sha1(s.encode()).hexdigest()

def catalyst(text:str)->str:
    t = text.lower()
    if any(k in t for k in ["earnings","results","eps","revenue","profit","loss"]): return "EARNINGS"
    if any(k in t for k in ["guidance","outlook","forecast"]): return "GUIDANCE"
    if any(k in t for k in ["acquire","acquisition","merger","buyout"]): return "M&A"
    if any(k in t for k in ["approval","sec","fda","investigation","fine","lawsuit","regulator"]): return "REGULATORY"
    if any(k in t for k in ["launch","product","partnership","contract"]): return "PRODUCT"
    if any(k in t for k in ["ceo","cfo","resigns","appoints","appointment"]): return "MGMT_CHANGE"
    return "OTHER"

def score(sent:float, cat:str):
    # sent in [-1,1]; simple transparent scoring
    cat_w = {"GUIDANCE":18,"EARNINGS":15,"M&A":22,"REGULATORY":20,"PRODUCT":10,"MGMT_CHANGE":8,"OTHER":4}
    base = 50
    t1 = max(0, min(100, base + 25*sent + cat_w.get(cat,4)))
    t5 = max(0, min(100, t1 + (5 if cat in {"GUIDANCE","REGULATORY","M&A"} else 0)))
    return int(round(t1)), int(round(t5))

def finnhub_news(ticker, token):
    fr = (datetime.utcnow() - timedelta(days=DAYS_BACK)).date().isoformat()
    to = datetime.utcnow().date().isoformat()
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={fr}&to={to}&token={token}"
    try:
        r = requests.get(url, timeout=20); r.raise_for_status()
        for x in r.json():
            ts = datetime.utcfromtimestamp(x.get("datetime",0)).isoformat()
            yield {
                "ticker": ticker,
                "source": "FINNHUB",
                "title": x.get("headline",""),
                "url": x.get("url",""),
                "published_at": ts,
                "snippet": (x.get("summary") or "")[:400]
            }
    except Exception:
        return

def edgar_rss(ticker):
    # SEC EDGAR Atom feed by ticker (CIK or symbol works reasonably for large caps)
    feed = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=&owner=exclude&count=40&output=atom"
    d = feedparser.parse(feed)
    for e in d.entries[:40]:
        yield {
            "ticker": ticker,
            "source": "EDGAR",
            "title": e.title,
            "url": e.link,
            "published_at": e.get("published",""),
            "snippet": (e.get("summary","") or "")[:400]
        }

# ---- APP ----
st.set_page_config(page_title="Stock Scout (Free)", layout="wide")
st.title("Stock Scout – High-Potential Signals (Free)")

token = os.getenv("FINNHUB_TOKEN", "").strip()
if not token:
    st.warning("Add your FINNHUB_TOKEN in Streamlit Secrets to enable company news.")
an = SentimentIntensityAnalyzer()

rows = []
seen = set()
for t in WATCHLIST:
    # Finnhub (if key present)
    if token:
        for it in finnhub_news(t, token):
            key = sha(it["source"]+it["title"]+it["published_at"])
            if key in seen: continue
            seen.add(key); rows.append(it)
    # SEC EDGAR
    for it in edgar_rss(t):
        key = sha(it["source"]+it["title"]+it["published_at"])
        if key in seen: continue
        seen.add(key); rows.append(it)

# Build DataFrame
if rows:
    df = pd.DataFrame(rows)
    # sentiment + catalyst + scores
    texts = (df["title"].fillna("") + ". " + df["snippet"].fillna("")).str.slice(0, 800)
    df["sentiment"] = [an.polarity_scores(x)["compound"] for x in texts]
    df["catalyst"] = [catalyst(x) for x in texts]
    sc = [score(s,c) for s,c in zip(df["sentiment"], df["catalyst"])]
    df["T+1"] = [x[0] for x in sc]
    df["T+5"] = [x[1] for x in sc]
    df.sort_values(["T+1","published_at"], ascending=[False, False], inplace=True)

    # Filters
    tickers = sorted(df["ticker"].unique())
    ft = st.multiselect("Filter tickers", tickers, default=[])
    min_t1 = st.slider("Min T+1 score", 0, 100, 60)
    cats = st.multiselect("Catalysts", sorted(df["catalyst"].unique()), default=[])

    q = df.copy()
    if ft:   q = q[q["ticker"].isin(ft)]
    if cats: q = q[q["catalyst"].isin(cats)]
    q = q[q["T+1"] >= min_t1]

    st.dataframe(q[["published_at","ticker","catalyst","T+1","T+5","title","source","url"]]
                 .rename(columns={"url":"source_link"}).reset_index(drop=True),
                 use_container_width=True)
else:
    st.info("No items yet. Try adding more tickers or wait a few minutes for feeds to populate.")
