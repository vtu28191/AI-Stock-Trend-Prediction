import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Stock Trend Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

    .main {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .title {
        font-size: 38px;
        font-weight: 700;
        color: #172033;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
    }

    .prediction-card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border: 1px solid #e5e7eb;
    }

    .prediction-title {
        font-size: 16px;
        color: #6b7280;
    }

    .prediction-value {
        font-size: 42px;
        font-weight: 800;
        margin-top: 5px;
    }

    .positive {
        color: #16a34a;
    }

    .negative {
        color: #dc2626;
    }

    .neutral {
        color: #d97706;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #172033;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    .news-card {
        background: white;
        padding: 16px;
        margin-bottom: 10px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }

    .footer {
        text-align: center;
        color: #777;
        font-size: 13px;
        margin-top: 40px;
        padding: 20px;
    }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("📊 Stock Predictor")

st.sidebar.markdown("### Select Stock")

stock = st.sidebar.selectbox(
    "Stock Symbol",
    ["AAPL", "TSLA", "MSFT", "AMZN", "GOOGL"]
)

st.sidebar.markdown("### Date Range")

days = st.sidebar.slider(
    "Historical Days",
    min_value=30,
    max_value=365,
    value=120
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Model")

st.sidebar.info(
    """
    **Prediction Model**

    🧠 LSTM  
    📰 FinBERT  
    📊 Time-Series Analysis  
    🤖 Machine Learning
    """
)

predict_button = st.sidebar.button(
    "🔮 Predict Trend",
    use_container_width=True
)


# ---------------------------------------------------------
# DEMO DATA
# ---------------------------------------------------------

np.random.seed(42)

dates = pd.date_range(
    end=datetime.today(),
    periods=days,
    freq="D"
)

base_prices = {
    "AAPL": 220,
    "TSLA": 250,
    "MSFT": 500,
    "AMZN": 230,
    "GOOGL": 250
}

base = base_prices[stock]

changes = np.random.normal(
    loc=0.001,
    scale=0.018,
    size=days
)

prices = base * np.cumprod(1 + changes)

high = prices * (1 + np.random.uniform(0, 0.02, days))
low = prices * (1 - np.random.uniform(0, 0.02, days))
open_price = prices * (1 + np.random.uniform(-0.01, 0.01, days))
volume = np.random.randint(
    1000000,
    8000000,
    days
)

data = pd.DataFrame({
    "Date": dates,
    "Open": open_price,
    "High": high,
    "Low": low,
    "Close": prices,
    "Volume": volume
})

# ---------------------------------------------------------
# TECHNICAL INDICATORS
# ---------------------------------------------------------

data["SMA"] = data["Close"].rolling(20).mean()

delta = data["Close"].diff()

gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

data["RSI"] = 100 - (100 / (1 + rs))

data = data.dropna().reset_index(drop=True)


# ---------------------------------------------------------
# DEMO PREDICTION
# ---------------------------------------------------------

current_price = data["Close"].iloc[-1]
previous_price = data["Close"].iloc[-2]

price_change = (
    (current_price - previous_price)
    / previous_price
) * 100

# Demo prediction logic
if price_change > 0.4:
    prediction = "UP"
    confidence = 86
    prediction_class = "positive"

elif price_change < -0.4:
    prediction = "DOWN"
    confidence = 82
    prediction_class = "negative"

else:
    prediction = "NEUTRAL"
    confidence = 76
    prediction_class = "neutral"


# ---------------------------------------------------------
# SENTIMENT DEMO
# ---------------------------------------------------------

sentiments = [
    ("Company reports strong quarterly revenue growth", "Positive", 0.82),
    ("Analysts discuss future market expansion", "Positive", 0.67),
    ("Market faces uncertainty due to global conditions", "Neutral", 0.51),
    ("Investors monitor upcoming economic decisions", "Neutral", 0.48),
]

avg_sentiment = np.mean(
    [x[2] for x in sentiments]
)

if avg_sentiment > 0.6:
    overall_sentiment = "Positive"
elif avg_sentiment < 0.4:
    overall_sentiment = "Negative"
else:
    overall_sentiment = "Neutral"


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="title">📈 AI-Powered Stock Market Trend Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Time-Series Analysis + Financial News Sentiment Analysis'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# TOP METRICS
# ---------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Current Price",
        f"${current_price:.2f}",
        f"{price_change:.2f}%"
    )

with col2:
    st.metric(
        "Predicted Trend",
        prediction
    )

with col3:
    st.metric(
        "Confidence",
        f"{confidence}%"
    )

with col4:
    st.metric(
        "News Sentiment",
        overall_sentiment
    )


# ---------------------------------------------------------
# PREDICTION CARD
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🔮 AI Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="prediction-card">

        <div class="prediction-title">
            Next Trading Day Trend
        </div>

        <div class="prediction-value {prediction_class}">
            {prediction}
        </div>

        <div style="color:#777; margin-top:10px;">
            Model Confidence: <b>{confidence}%</b>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# STOCK PRICE CHART
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">📊 Historical Stock Price</div>',
    unsafe_allow_html=True
)

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    data["Date"],
    data["Close"],
    label="Close Price"
)

ax.plot(
    data["Date"],
    data["SMA"],
    label="SMA (20)"
)

ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title(f"{stock} Historical Price & SMA")

ax.legend()

ax.grid(alpha=0.2)

st.pyplot(fig)


# ---------------------------------------------------------
# TECHNICAL INDICATORS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">📐 Technical Indicators</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    fig1, ax1 = plt.subplots(figsize=(7, 3))

    ax1.plot(
        data["Date"],
        data["SMA"]
    )

    ax1.set_title("Simple Moving Average (SMA)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("SMA")

    ax1.grid(alpha=0.2)

    st.pyplot(fig1)


with col2:

    fig2, ax2 = plt.subplots(figsize=(7, 3))

    ax2.plot(
        data["Date"],
        data["RSI"]
    )

    ax2.axhline(70)
    ax2.axhline(30)

    ax2.set_title("Relative Strength Index (RSI)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("RSI")

    ax2.grid(alpha=0.2)

    st.pyplot(fig2)


# ---------------------------------------------------------
# SENTIMENT ANALYSIS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">📰 Financial News Sentiment Analysis</div>',
    unsafe_allow_html=True
)

sent_col1, sent_col2, sent_col3 = st.columns(3)

with sent_col1:
    st.metric(
        "Overall Sentiment",
        overall_sentiment
    )

with sent_col2:
    st.metric(
        "Sentiment Score",
        f"{avg_sentiment:.2f}"
    )

with sent_col3:
    st.metric(
        "News Analysed",
        len(sentiments)
    )


# ---------------------------------------------------------
# NEWS
# ---------------------------------------------------------

st.markdown("### Latest Financial News")

for headline, sentiment, score in sentiments:

    if sentiment == "Positive":
        icon = "🟢"
    elif sentiment == "Negative":
        icon = "🔴"
    else:
        icon = "🟡"

    st.markdown(
        f"""
        <div class="news-card">

            <b>{headline}</b>

            <br><br>

            {icon} <b>{sentiment}</b>

            &nbsp;&nbsp;

            Sentiment Score: {score:.2f}

        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# MODEL PIPELINE
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🧠 AI Model Pipeline</div>',
    unsafe_allow_html=True
)

pipeline = st.columns(7)

steps = [
    "Stock Data",
    "Preprocessing",
    "Features",
    "FinBERT",
    "Combination",
    "LSTM",
    "Prediction"
]

for col, step in zip(pipeline, steps):

    with col:

        st.info(step)


# ---------------------------------------------------------
# DATA TABLE
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">📋 Latest Stock Data</div>',
    unsafe_allow_html=True
)

display_data = data.tail(10).copy()

display_data["Date"] = display_data["Date"].dt.strftime(
    "%Y-%m-%d"
)

display_data["Close"] = display_data["Close"].round(2)
display_data["SMA"] = display_data["SMA"].round(2)
display_data["RSI"] = display_data["RSI"].round(2)

st.dataframe(
    display_data,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">⚙️ Model Information</div>',
    unsafe_allow_html=True
)

info1, info2, info3 = st.columns(3)

with info1:
    st.markdown(
        """
        **LSTM**

        Used for learning temporal patterns
        from historical stock sequences.
        """
    )

with info2:
    st.markdown(
        """
        **FinBERT**

        Used for financial-news sentiment
        classification.
        """
    )

with info3:
    st.markdown(
        """
        **Output**

        UP / DOWN / NEUTRAL prediction
        with confidence information.
        """
    )


# ---------------------------------------------------------
# DISCLAIMER
# ---------------------------------------------------------

st.warning(
    "⚠️ This dashboard is intended for analysis and "
    "educational purposes. It does not guarantee future "
    "stock-market movements and should not be treated as "
    "financial advice."
)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">

        AI-Powered Stock Market Trend Prediction<br>

        LSTM + FinBERT + Time-Series Analysis + Sentiment Analysis

    </div>
    """,
    unsafe_allow_html=True
)