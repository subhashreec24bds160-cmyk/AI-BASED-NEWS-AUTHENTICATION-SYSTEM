import streamlit as st

st.set_page_config(
    page_title="TruthScope AI",
    page_icon="🔍",
    layout="wide"
)

# ------------------ CSS ------------------

st.markdown("""
<style>

.main{
    background-color:#f8fbff;
}

.title{
    text-align:center;
    font-size:60px;
    font-weight:bold;
    color:#0b3d91;
}

.subtitle{
    text-align:center;
    font-size:22px;
    color:gray;
}

.feature{
    background:#ffffff;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 2px 10px rgba(0,0,0,0.15);
    text-align:center;
    margin:10px;
}

.footer{
    text-align:center;
    color:gray;
    padding-top:50px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------

st.markdown("<div class='title'>🔍 TruthScope AI</div>", unsafe_allow_html=True)

st.markdown(
"<div class='subtitle'>Know the Truth Behind Every Story</div>",
unsafe_allow_html=True
)

st.write("")
st.write("")

st.markdown(
"""
<center>

### AI-Based News Authentication System

</center>
""",
unsafe_allow_html=True
)

st.write("")
st.write("")

# ---------------- Features ----------------

col1,col2,col3=st.columns(3)

with col1:
    st.info("📰 **Text News Detection**")

with col2:
    st.info("🖼 **Image News Detection**")

with col3:
    st.info("🤖 **Google Gemini AI Summary & AI Explanation**")

col4,col5,col6=st.columns(3)

with col4:
    st.success("📊 Fake News Percentage")

with col5:
    st.success("🛡 Source Credibility")

with col6:
    st.success("😊 Sentiment Analysis")

st.write("")
st.write("")

st.markdown("---")

st.write("")

# --------------- Start Button ------------

if st.button("🚀 Start Detection", use_container_width=True):
    st.switch_page("pages/analyzer.py")

st.markdown(
"""
<div class='footer'>

Made with ❤️ using Streamlit, Machine Learning & Google Gemini

</div>
""",
unsafe_allow_html=True
)