# streamlit run agent_app.py
# pip install streamlit google-genai requests beautifulsoup4 lxml pandas python-dateutil pytrends fredapi

import os, json, time, requests
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil import parser as dtp

# --- LLM (Gemini) ---
from google import genai

# --- Trends and macro ---
from pytrends.request import TrendReq
from fredapi import Fred

st.set_page_config(page_title="AI News & Signals Agent", layout="wide")

# Secrets
APP_KEY = st.secrets.get("APP_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
CSE_API_KEY = st.secrets.get("CSE_API_KEY", "")
CSE_CX = st.secrets.get("CSE_CX", "")
FRED_API_KEY = st.secrets.get("FRED_API_KEY", "")

# Regulatory sources (always-on)
SEBI_RSS = "https://www.sebi.gov.in/sebirss.xml"
DGFT_NOTICES = "https://www.dgft.gov.in/CP/?opt=notification"

# Helpers
def iso(dt_str):
    try:
        return dtp.parse(dt_str).isoformat()
    except Exception:
        return datetime.utcnow().isoformat()

@st.cache_data(ttl=900)
def fetch_rss(url, limit=30):
    import feedparser
    d = feedparser.parse(url)
    rows = []
    for e in d.entries[:limit]:
        rows.append({
            "category": "Regulatory",
            "source": getattr(d.feed, "title", url),
            "title": getattr(e, "title", ""),
            "link": getattr(e, "link", ""),
            "summary": getattr(e, "summary", getattr(e, "description", "")),
            "published": iso(getattr(e, "published", getattr(e, "updated", ""))),
        })
    return pd.DataFrame(rows)

def gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)

def gemini_expand_queries(topic: str) -> list:
    client = gemini_client()
    prompt = f"""
Generate 5 high-signal web search queries for: "{topic}".
Blend world current affairs, investment/valuation (P/E, P/B, expected return, CAPM), SEBI & DGFT/regulatory, and martech/social signals.
Return a JSON list of strings; no prose.
"""
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    try:
        return json.loads(resp.text)
    except Exception:
        return [topic]

@st.cache_data(ttl=600)
def cse_search(query: str, api_key: str, cx: str, num=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cx, "q": query, "num": num, "safe": "off"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    items = js.get("items", []) or []
    return [{"title": it.get("title",""), "link": it.get("link",""), "snippet": it.get("snippet","")} for it in items]

def fetch_and_extract(url: str) -> dict:
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0 Agent"}, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script","style","nav","header","footer","aside"]): tag.decompose()
        article = soup.find("article")
        text = " ".join(p.get_text(" ", strip=True) for p in (article or soup).find_all(["p","li"]))[:8000]
        return {"ok": True, "text": text}
    except Exception as e:
        return {"ok": False, "error": str(e), "text": ""}

def trends_timeseries(keyword: str, geo="IN"):
    py = TrendReq()
    py.build_payload([keyword], timeframe="now 7-d", geo=geo)
    df = py.interest_over_time().reset_index()
    if "isPartial" in df: df = df.drop(columns=["isPartial"])
    return df

def fred_series(series_id: str):
    fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else Fred()
    s = fred.get_series(series_id)
    return s.reset_index(names=["date"]).rename(columns={0:"value"})

# UI: gate
with st.form("gate"):
    key_in = st.text_input("Enter access key", type="password")
    topic = st.text_input("Topic (e.g., SEBI insider trading rules impact on fintech valuations)")
    submitted = st.form_submit_button("Run Agent")

if not submitted:
    st.info("Submit a topic and key to run the agent.", icon="🔐")
    st.stop()

if key_in != APP_KEY:
    st.error("Access key invalid.", icon="⛔")
    st.stop()

# Run pipeline
tabs = st.tabs(["Chat & URLs","Scraped Data","Regulatory Watch","Trends & Macro"])

with tabs[0]:
    st.subheader("Chat with Agent")
    st.caption("Type follow‑ups below; first, the agent expands queries and finds URLs.")
    with st.status("Generating search queries with Gemini...", expanded=False):
        queries = gemini_expand_queries(topic)
        st.write(queries)
    urls = []
    with st.status("Searching the web (Programmable Search)...", expanded=False):
        for q in queries:
            urls += cse_search(q, CSE_API_KEY, CSE_CX, num=5)
        url_df = pd.DataFrame(urls).drop_duplicates(subset=["link"])
        st.write(url_df)
    st.session_state["url_df"] = url_df

    # simple chat
    user_prompt = st.chat_input("Ask about the topic or the found URLs")
    if user_prompt and GEMINI_API_KEY:
        with st.chat_message("user"): st.write(user_prompt)
        with st.chat_message("assistant"):
            client = gemini_client()
            context = "\n".join(f"- {t} :: {l}" for t,l in url_df[["title","link"]].head(10).values)
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"Context URLs:\n{context}\n\nUser: {user_prompt}\nGive concise, referenced guidance."
            )
            st.write(resp.text)

with tabs[1]:
    st.subheader("Scraped Content")
    if "url_df" not in st.session_state or st.session_state["url_df"].empty:
        st.warning("No URLs yet.")
    else:
        df = st.session_state["url_df"].copy()
        results = []
        prog = st.progress(0)
        for i, row in enumerate(df.itertuples(), 1):
            ext = fetch_and_extract(row.link)
            results.append({
                "title": row.title, "link": row.link, "snippet": row.snippet,
                "ok": ext["ok"], "len": len(ext["text"]), "text": ext["text"][:1200]
            })
            prog.progress(i/len(df))
        out = pd.DataFrame(results)
        st.dataframe(out[["title","len","ok","snippet","link"]], use_container_width=True)
        st.session_state["scraped"] = out

with tabs[2]:
    st.subheader("Regulatory Watch (SEBI + DGFT)")
    left, right = st.columns(2)
    with left:
        sebi = fetch_rss(SEBI_RSS, limit=25)
        st.write("SEBI latest")
        st.dataframe(sebi[["title","published","link"]], use_container_width=True)
    with right:
        st.write("DGFT notifications portal")
        st.link_button("Open DGFT notifications", DGFT_NOTICES)

with tabs[3]:
    st.subheader("Social Trends & Economic Indicators")
    k = topic.split()[0]
    try:
        tr = trends_timeseries(k, geo="IN")
        st.line_chart(tr.set_index("date")[k])
    except Exception as e:
        st.warning(f"Trends error: {e}")
    try:
        # Example macro: CPIAUCSL or GDP; replace per need
        macro_id = st.selectbox("FRED series", ["CPIAUCSL","FEDFUNDS","GDP"], index=0)
        macro = fred_series(macro_id)
        st.line_chart(macro.set_index("date")["value"])
    except Exception as e:
        st.warning(f"FRED error: {e}")
