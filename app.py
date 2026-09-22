import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Breakout Tracker + Deep Dive", layout="wide")
st.title("📈 Stock Breakout Tracker & Deep Dive Analysis")
st.caption(f"Last run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

DEFAULT_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "ITC.NS", "LT.NS", "BAJFINANCE.NS", "MARUTI.NS"
]

GLOBAL_INDICES = {
    "Nifty 50 (India)": "^NSEI",
    "Sensex (India)": "^BSESN",
    "S&P 500 (US)": "^GSPC",
    "Nasdaq (US)": "^IXIC",
    "Dow Jones (US)": "^DJI",
    "STI (Singapore)": "^STI",
    "Nikkei 225 (Japan)": "^N225",
    "Hang Seng (Hong Kong)": "^HSI",
    "FTSE 100 (UK)": "^FTSE",
}

with st.sidebar:
    st.header("Scanner Settings")
    tickers_input = st.text_area("Tickers (comma separated)", value=", ".join(DEFAULT_TICKERS), height=120)
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    threshold_pct = st.slider("Near-ATH threshold (%)", 0.5, 10.0, 3.0, 0.5)
    lookback_years = st.selectbox("Data lookback", ["1y", "2y", "5y", "max"], index=1)
    run_button = st.button("🔍 Run Scan", type="primary")

tab1, tab2 = st.tabs(["📊 Scanner", "🔬 Deep Dive Analysis"])

# ============================================================
# CANDLESTICK PATTERN DETECTION (rule-based, no external lib)
# ============================================================
def detect_patterns(df):
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = (c - o).abs()
    range_ = (h - l).replace(0, np.nan)
    upper_wick = h - df[["Open", "Close"]].max(axis=1)
    lower_wick = df[["Open", "Close"]].min(axis=1) - l

    patterns = pd.DataFrame(index=df.index)

    # Doji: body is very small relative to range
    patterns["Doji"] = (body / range_) < 0.1

    # Hammer: small body near top, long lower wick, small upper wick
    patterns["Hammer"] = (lower_wick > 2 * body) & (upper_wick < body) & (body / range_ < 0.35)

    # Bullish Engulfing
    prev_o, prev_c = o.shift(1), c.shift(1)
    patterns["Bullish_Engulfing"] = (prev_c < prev_o) & (c > o) & (o <= prev_c) & (c >= prev_o)

    # Bearish Engulfing
    patterns["Bearish_Engulfing"] = (prev_c > prev_o) & (c < o) & (o >= prev_c) & (c <= prev_o)

    # Morning Star (3-candle bullish reversal)
    c2, o2 = c.shift(2), o.shift(2)
    body1 = (c2 - o2).abs()
    small_mid = body.shift(1) / range_.shift(1) < 0.3
    patterns["Morning_Star"] = (c2 < o2) & small_mid & (c > o) & (c > (o2 + c2) / 2)

    # Evening Star (3-candle bearish reversal)
    patterns["Evening_Star"] = (c2 > o2) & small_mid & (c < o) & (c < (o2 + c2) / 2)

    return patterns.fillna(False)


def backtest_pattern(df, pattern_mask, forward_days=(5, 10, 20)):
    results = {}
    idx_positions = np.where(pattern_mask.values)[0]
    for fd in forward_days:
        rets = []
        for pos in idx_positions:
            if pos + fd < len(df):
                start_price = df["Close"].iloc[pos]
                end_price = df["Close"].iloc[pos + fd]
                rets.append((end_price - start_price) / start_price * 100)
        if rets:
            rets = np.array(rets)
            results[fd] = {
                "occurrences": len(rets),
                "win_rate": round((rets > 0).mean() * 100, 1),
                "avg_return": round(rets.mean(), 2),
                "best": round(rets.max(), 2),
                "worst": round(rets.min(), 2),
            }
        else:
            results[fd] = {"occurrences": 0, "win_rate": None, "avg_return": None, "best": None, "worst": None}
    return results


def scan_stock(ticker, period, threshold_pct):
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty or len(data) < 30:
            return None
        current_price = float(data["Close"].iloc[-1])
        all_time_high = float(data["High"].max())
        pct_from_high = (current_price - all_time_high) / all_time_high * 100

        def pct_return(days):
            if len(data) > days:
                past_price = float(data["Close"].iloc[-days])
                return round((current_price - past_price) / past_price * 100, 2)
            return None

        avg_vol_20 = float(data["Volume"].iloc[-21:-1].mean())
        today_vol = float(data["Volume"].iloc[-1])
        vol_spike = round(today_vol / avg_vol_20, 2) if avg_vol_20 > 0 else None

        if pct_from_high >= -threshold_pct:
            return {
                "Ticker": ticker, "Current Price": round(current_price, 2),
                "All-Time High": round(all_time_high, 2), "% From ATH": round(pct_from_high, 2),
                "Status": "🚀 Breakout" if pct_from_high >= 0 else "🔶 Near ATH",
                "1W Return %": pct_return(5), "1M Return %": pct_return(22), "3M Return %": pct_return(66),
                "Volume Spike (x avg)": vol_spike,
            }
        return None
    except Exception as e:
        st.warning(f"Skipped {ticker}: {e}")
        return None


# ============================================================
# TAB 1: SCANNER
# ============================================================
with tab1:
    if run_button:
        results = []
        progress = st.progress(0, text="Scanning stocks...")
        for i, ticker in enumerate(tickers):
            result = scan_stock(ticker, lookback_years, threshold_pct)
            if result:
                results.append(result)
            progress.progress((i + 1) / len(tickers), text=f"Scanning {ticker}...")
        progress.empty()

        if results:
            df = pd.DataFrame(results).sort_values("% From ATH", ascending=False)
            st.session_state["scan_results"] = df
            st.success(f"Found {len(df)} stock(s) near/at all-time high")
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, "breakout_results.csv", "text/csv")
        else:
            st.info("No stocks matched. Try increasing the threshold %.")
    else:
        st.info("👈 Set tickers and threshold in the sidebar, then click **Run Scan**.")
        if "scan_results" in st.session_state:
            st.dataframe(st.session_state["scan_results"], use_container_width=True, hide_index=True)

# ============================================================
# TAB 2: DEEP DIVE
# ============================================================
with tab2:
    st.subheader("Deep Dive: Candlestick Analysis, Pattern Backtest, News & Global Comparison")
    deep_ticker = st.text_input("Enter a ticker to analyze", value=tickers[0] if tickers else "RELIANCE.NS")
    deep_period = st.selectbox("Chart period", ["6mo", "1y", "2y", "5y"], index=1, key="deep_period")
    analyze_btn = st.button("🔬 Run Deep Dive", type="primary")

    if analyze_btn and deep_ticker:
        with st.spinner(f"Analyzing {deep_ticker}..."):
            data = yf.download(deep_ticker, period=deep_period, progress=False)

            if data.empty:
                st.error("No data found for this ticker. Check the symbol.")
            else:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                # --- Candlestick chart ---
                st.markdown("### 🕯️ Candlestick Chart")
                fig = go.Figure(data=[go.Candlestick(
                    x=data.index, open=data["Open"], high=data["High"],
                    low=data["Low"], close=data["Close"], name=deep_ticker
                )])
                fig.update_layout(xaxis_rangeslider_visible=False, height=500,
                                   margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

                # --- Pattern detection + backtest ---
                st.markdown("### 🔍 Candlestick Pattern Backtest (Historical Win Rate)")
                st.caption(
                    "This shows how often each pattern appeared historically for this stock, and what "
                    "return followed over the next 5/10/20 trading days. This is a statistical backtest "
                    "on past data only — it does not predict future results."
                )
                patterns = detect_patterns(data)
                pattern_summary = []
                for pname in patterns.columns:
                    mask = patterns[pname]
                    count = int(mask.sum())
                    if count == 0:
                        continue
                    bt = backtest_pattern(data, mask)
                    for fd, stats in bt.items():
                        if stats["occurrences"] > 0:
                            pattern_summary.append({
                                "Pattern": pname.replace("_", " "),
                                "Times Seen": count,
                                "Forward Window": f"{fd}d",
                                "Win Rate %": stats["win_rate"],
                                "Avg Return %": stats["avg_return"],
                                "Best %": stats["best"],
                                "Worst %": stats["worst"],
                            })
                if pattern_summary:
                    pdf = pd.DataFrame(pattern_summary).sort_values(["Pattern", "Forward Window"])
                    st.dataframe(pdf, use_container_width=True, hide_index=True)

                    # Most recent pattern signal
                    last_row = patterns.iloc[-1]
                    active = [p.replace("_", " ") for p in patterns.columns if last_row[p]]
                    if active:
                        st.success(f"📌 Pattern detected on latest candle: **{', '.join(active)}**")
                    else:
                        st.info("No recognized pattern on the most recent candle.")
                else:
                    st.info("No patterns detected in this period — try a longer lookback.")

                # --- News ---
                st.markdown("### 📰 Recent News")
                try:
                    ticker_obj = yf.Ticker(deep_ticker)
                    news_items = ticker_obj.news[:6] if ticker_obj.news else []
                    if news_items:
                        for item in news_items:
                            n = item.get("content", item)
                            title = n.get("title", "Untitled")
                            link = n.get("canonicalUrl", {}).get("url", "") if isinstance(n.get("canonicalUrl"), dict) else n.get("link", "")
                            pub = n.get("provider", {}).get("displayName", "") if isinstance(n.get("provider"), dict) else n.get("publisher", "")
                            if link:
                                st.markdown(f"- [{title}]({link})  \n  *{pub}*")
                            else:
                                st.markdown(f"- {title}  \n  *{pub}*")
                    else:
                        st.info("No recent news found via Yahoo Finance for this ticker.")
                except Exception as e:
                    st.info("News unavailable right now.")

                # --- Global market comparison ---
                st.markdown("### 🌏 Global Market Comparison")
                st.caption(
                    "Normalized % change comparison — shows if this stock is outperforming or lagging "
                    "major world indices, including Singapore's STI (SGX Nifty futures are no longer "
                    "listed on public feeds since moving to GIFT City in 2022)."
                )
                selected_indices = st.multiselect(
                    "Compare against:",
                    options=list(GLOBAL_INDICES.keys()),
                    default=["Nifty 50 (India)", "S&P 500 (US)", "STI (Singapore)"]
                )

                if selected_indices:
                    comp_fig = go.Figure()
                    stock_norm = (data["Close"] / data["Close"].iloc[0] - 1) * 100
                    comp_fig.add_trace(go.Scatter(x=data.index, y=stock_norm, name=deep_ticker, line=dict(width=3)))

                    for name in selected_indices:
                        idx_ticker = GLOBAL_INDICES[name]
                        try:
                            idx_data = yf.download(idx_ticker, period=deep_period, progress=False)
                            if not idx_data.empty:
                                if isinstance(idx_data.columns, pd.MultiIndex):
                                    idx_data.columns = idx_data.columns.get_level_values(0)
                                idx_norm = (idx_data["Close"] / idx_data["Close"].iloc[0] - 1) * 100
                                comp_fig.add_trace(go.Scatter(x=idx_data.index, y=idx_norm, name=name, line=dict(dash="dot")))
                        except Exception:
                            continue

                    comp_fig.update_layout(
                        height=450, yaxis_title="% Change",
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    st.plotly_chart(comp_fig, use_container_width=True)

st.divider()
st.caption(
    "Data via Yahoo Finance (yfinance). Pattern backtests and news are for informational/educational "
    "purposes only, not financial advice or a guarantee of future performance. Always do your own research."
)
