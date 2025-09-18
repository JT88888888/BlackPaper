import os, re, time, json, hashlib, random, requests, pandas as pd, feedparser, streamlit as st
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import openai
openai.api_key = st.secrets["OPENAI_API_KEY"]
FINNHUB_TOKEN = st.secrets["FINNHUB_TOKEN"]
# --- New dashboard header/navigation (B) ---
st.set_page_config(page_title="BlackPaper – Market Intelligence", layout="wide")

def inject_style():
    st.markdown("""
    <style>
      .app-header {display:flex; gap:14px; align-items:center; margin:10px 0 4px 0;}
      .app-title {font-size:30px; font-weight:700;}
      .pill {padding:4px 8px; border-radius:999px; background:#0b5; color:white; font-size:12px;}
      .nav {display:flex; gap:18px; margin:6px 0 14px 0; opacity:.95; position:relative; z-index:10}
      .nav a {text-decoration:none; color:#9ca3af; font-weight:600}
      .nav a.active {color:#fff; border-bottom:2px solid #1abc9c; padding-bottom:4px}
      .card {background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
             border-radius:14px; padding:14px 16px;}
      .kpi-title {font-size:13px; color:#9ca3af; margin-bottom:6px}
      .kpi-value {font-size:24px; font-weight:700}
      .kpi-delta-up {color:#22c55e; font-size:12px}
      .kpi-delta-down {color:#ef4444; font-size:12px}
      .section-title {font-size:18px; font-weight:700; margin:10px 0}
      .muted {color:#9ca3af; font-size:12px}
      table td, table th {font-size: 13px;}
    </style>
    """, unsafe_allow_html=True)

inject_style()

st.markdown("""
<div class="app-header">
  <div class="app-title">Market Intelligence Dashboard</div>
  <span class="pill">AI-powered</span>
</div>
<div class="nav">
  <a class="active" href="#dash">Dashboard</a>
  <a href="#signals">Top Signals</a>
  <a href="#analyst88">Analyst88</a>
  <a href="#settings">Settings</a>
</div>
""", unsafe_allow_html=True)
# --- End new header/navigation ---
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
st.markdown('<a id="dash"></a>', unsafe_allow_html=True)
st.subheader("Auto Top Picks")
picks = df[(df["T+1"]>=auto_min_t1) & (df["confidence"]>=auto_min_conf)].copy()
st.write(f"{len(picks)} picks (T+1 ≥ {auto_min_t1}, confidence ≥ {auto_min_conf}) from a batch of {len(batch)} symbols")
st.dataframe(
    picks[["published_at","ticker","catalyst","polarity","confidence","T+1","T+5","title","source","url"]]
    .rename(columns={"url":"source_link"}).reset_index(drop=True),
    use_container_width=True
)

# ---- Full feed + filters ----
st.markdown('<a id="signals"></a>', unsafe_allow_html=True)
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
# =======================
# Analyst88 (OpenAI pass)
# =======================
import json
from math import ceil

try:
    from openai import OpenAI
    _openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    _openai_ready = bool(_openai_key)
except Exception:
    _openai_ready = False
st.markdown('<a id="analyst88"></a>', unsafe_allow_html=True)
st.subheader("Analyst88 – AI Flags (next 1–5 days)")

with st.expander("Run Analyst88 on the feed (uses your OpenAI API key)"):
    colA, colB, colC = st.columns(3)
    with colA:
        min_t1_gate = st.slider("Pre-filter by T+1 (to cut noise)", 0, 100, 70)
    with colB:
        max_items = st.slider("Max items to analyze this run", 50, 400, 200, step=50)
    with colC:
        model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0,
                                  help="Mini is cheaper; 4o is stronger but costs more.")

    if not _openai_ready:
        st.warning("Add OPENAI_API_KEY in **Manage app → Secrets** to enable Analyst88.")
    else:
        # Build a compact, token-friendly list of items for the LLM
        df_aa = df[df["T+1"] >= min_t1_gate].copy()
        if df_aa.empty:
            st.info("Nothing passes the pre-filter. Lower the T+1 gate or wait for new items.")
        else:
            # Use a stable id so the LLM can reference/return items
            def _mkid(row):
                return sha(f'{row["source"]}|{row["title"]}|{row["published_at"]}')
            df_aa["aa_id"] = df_aa.apply(_mkid, axis=1)

            # Order by recency then T+1 and clip
            df_aa.sort_values(["published_at","T+1"], ascending=[False, False], inplace=True)
            df_aa = df_aa.head(max_items)

            # Prepare payload (shorten text for token safety)
            def _shorten(text, n=350):
                return (text or "")[:n]
            items = []
            for _, r in df_aa.iterrows():
                items.append({
                    "id": r["aa_id"],
                    "ticker": r["ticker"],
                    "catalyst": r["catalyst"],
                    "t1": int(r["T+1"]),
                    "t5": int(r["T+5"]),
                    "polarity": float(r["polarity"]),
                    "confidence": float(r["confidence"]),
                    "title": _shorten(r["title"], 200),
                    "snippet": _shorten(str(r.get("snippet","")), 350),
                    "source": r["source"],
                    "published_at": str(r["published_at"]),
                    "url": r["url"],
                })

            st.write(f"Sending {len(items)} items to Analyst88…")

            # System & user prompts
            system_prompt = (
                "You are Analyst88, an equity event-driven analyst. "
                "Task: from the provided news/filing items (each with sentiment & catalyst), "
                "flag those most likely to see **upward** price impact in the **next 1–5 trading days**. "
                "Favor catalysts like GUIDANCE↑, REGULATORY approvals, M&A (acquirer/target context), "
                "material PRODUCT wins, positive EARNINGS surprises. Penalize vague/old items. "
                "Return strict JSON: an array of objects with keys:\n"
                "  id (from input), potential (0..100), horizon ('T+1' or 'T+5'), "
                "  rationale (<=240 chars), risk_notes (<=140 chars).\n"
                "Only include items you consider high potential (potential ≥ 80)."
            )
            user_prompt = (
                "Items JSON follows. Score and return only high-potential longs.\n"
                "INPUT_ITEMS_JSON:\n" + json.dumps(items, ensure_ascii=False)
            )

            if st.button("Run Analyst88 now", type="primary"):
                try:
                    client = OpenAI(api_key=_openai_key)
                    resp = client.chat.completions.create(
                        model=model_name,
                        temperature=0.2,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    raw = resp.choices[0].message.content
                    # Expecting {"high_potential":[...]} or just a list; handle both
                    try:
                        parsed = json.loads(raw)
                        hp = parsed.get("high_potential", parsed if isinstance(parsed, list) else [])
                    except Exception:
                        hp = []
                    if not hp:
                        st.info("Analyst88 returned no high-potential items for this batch.")
                    else:
                        # Merge back with df_aa for display
                        hp_df = pd.DataFrame(hp)
                        if "id" not in hp_df.columns:
                            st.warning("Unexpected response shape from Analyst88.")
                        else:
                            merged = hp_df.merge(df_aa, left_on="id", right_on="aa_id", how="left")
                            show_cols = ["published_at","ticker","catalyst","T+1","T+5",
                                         "potential","horizon","rationale","risk_notes",
                                         "title","source","url"]
                            st.success(f"Analyst88 flagged {len(merged)} items")
                            st.dataframe(
                                merged[show_cols].sort_values(["potential","T+1"], ascending=[False, False]).reset_index(drop=True),
                                use_container_width=True
                            )
                            st.caption("Note: Heuristic + LLM opinion. Research only, not investment advice.")
                            st.session_state["df_all"] = merged.copy()

                except Exception as e:
                    st.error(f"Analyst88 failed: {e}")
# --- UI STYLE PACK ---
def inject_style():
    import streamlit as st
    st.markdown("""
<style>
  .app-header {
    display:flex; gap:14px; align-items:center; margin:10px 0 4px 0;
    /* Add these two lines */
    position: relative;
    z-index: 1000;
  }

  .app-title {font-size:30px; font-weight:700;}

  .pill {padding:4px 8px; border-radius:999px; background:#0b5; color:white; font-size:12px;}

  /* UPDATE THIS RULE */
  .nav {
    display:flex; gap:18px; margin:6px 0 14px 0; opacity:.95;
    /* Make the nav layer sit above the rest */
    position: relative;
    z-index: 1001;
  }

  /* ADD THIS RULE so the links definitely accept clicks */
  .nav a {
    pointer-events: auto;
    position: relative;
    z-index: 1002;
  }

  .nav a {text-decoration:none; color:#9ca3af; font-weight:600}
  .nav a.active {color:#fff; border-bottom:2px solid #1abc9c; padding-bottom:4px}

  .card {background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
         border-radius:14px; padding:14px 16px}

  .kpi-title {font-size:13px; color:#9ca3af; margin-bottom:6px}
  .kpi-value {font-size:24px; font-weight:700}
  .kpi-delta-up {color:#22c55e; font-size:12px}
  .kpi-delta-down {color:#ef4444; font-size:12px}

  .section-title {font-size:18px; font-weight:700; margin:10px 0}
  .muted {color:#9ca3af; font-size:12px}
  table td, table th {font-size: 13px}
</style>
""", unsafe_allow_html=True)

# ---------------- Analyst88 -----------------
st.header("Analyst88 – AI Stock Screener")

st.write("Click below to have AI scan all current feeds for high-potential stocks and indexes.")
if st.button("Run Analyst88"):
    with st.spinner("AI is analysing all feeds and indexes..."):
        df_all = st.session_state.get("df_all")
        if df_all is None or df_all.empty:
            st.warning("No data available yet for Analyst88.")
            st.stop()

        # Pull the latest data table (you may already have a dataframe like 'df')
        # ...rest of your Analyst88 code...


            # Pull the latest data table (you may already have a dataframe like 'df')
            df_all = merged if 'merged' in locals() else None

            if df_all is not None:
                prompt = (
                    "Analyze this market data and flag tickers (including indexes) "
                    "that are most likely to rise in the next few days. "
                    "Explain briefly why for each pick.\n\n"
                    f"{df_all.to_csv(index=False)}"
                )
                completion = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": "You are a financial analyst."},
                              {"role": "user", "content": prompt}]
                )
                st.subheader("AI Recommendations")
                st.write(completion.choices[0].message["content"])
            else:
                st.warning("No data available yet for Analyst88.")
else:
    st.error("OPENAI_API_KEY not found. Add it to Streamlit secrets.")
st.markdown('<a id="settings"></a>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)
# --- Analyst88 section anchor and header ---
st.markdown('<a id="analyst88"></a>', unsafe_allow_html=True)
st.header("Analyst88")
st.write("AI-powered deep scan will appear here…")
# (You can expand this with your Analyst88 analysis logic)
if st.button("Run Analyst88 full scan"):
    st.info("Scanning latest signals…")
    # Example: reuse your aggregated data frame (adjust variable if different)
    all_items = df_all.to_dict(orient="records")

    system_prompt = "You are a financial analyst. From this JSON list, find and explain stocks with strong upside potential in the next few days."
    user_prompt = json.dumps(all_items)

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        st.write(completion.choices[0].message["content"])
    except Exception as e:
        st.error(f"Analyst88 scan failed: {e}")
if st.button("Refresh Ticker List (Admin)"):
    import requests, csv, os
    r = requests.get(
        f"https://finnhub.io/api/v1/stock/symbol?exchange=US&token={os.environ['FINNHUB_TOKEN']}"
    )
    data = r.json()
    with open("tickers.csv", "w", newline="") as f:
        w = csv.writer(f)
        for d in data:
            w.writerow([d["symbol"]])
    st.success(f"Updated tickers.csv with {len(data)} symbols")
