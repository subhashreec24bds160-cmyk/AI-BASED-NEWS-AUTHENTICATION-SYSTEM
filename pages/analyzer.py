import streamlit as st
import pandas as pd
import joblib
import re
import os
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import easyocr
from PIL import Image
import numpy as np

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="TruthScope AI",
    page_icon="🔍",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
.main {
    background: #f4f7fb;
}
.header-box {
    background: linear-gradient(90deg, #2563EB, #7C3AED);
    padding: 25px;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}
.result-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.15);
    margin-bottom: 20px;
}
.ai-card {
    background: white;
    padding: 18px;
    border-radius: 15px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.15);
    margin-bottom: 18px;
}
.metric-card {
    background: #ffffff;
    padding: 15px;
    border-radius: 12px;
    border-left: 6px solid #2563EB;
}
.footer {
    text-align: center;
    color: gray;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# GEMINI API SETUP
# ==========================================
api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

if not api_key:
    api_key = st.sidebar.text_input("🔑 Enter Gemini API Key", type="password")
    if not api_key:
        st.sidebar.warning("Please provide a Gemini API Key to unlock AI Insights.")

if api_key:
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    gemini_model = None

# ==========================================
# LOAD ML MODEL & DATASET
# ==========================================
@st.cache_resource
def load_model():
    model = joblib.load("label_model.pkl")
    vectorizer = joblib.load("truthscope_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

ocr_reader = load_ocr()

@st.cache_data
def load_dataset():
    df = pd.read_excel(r"C:\Users\DELL\Downloads\fav dataset.xlsx")
    df["combined_text"] = (
        df["headline"].astype(str)
        + " "
        + df["body text"].astype(str)
    )
    return df

df = load_dataset()
dataset_vectors = vectorizer.transform(df["combined_text"])

# ==========================================
# HEADER
# ==========================================
st.markdown("""
<div class="header-box">
    <h1>🔍 TruthScope AI</h1>
    <h4>Know the Truth Behind Every Story</h4>
    AI Powered Fake News Detection System
</div>
""", unsafe_allow_html=True)

# ==========================================
# TWO COLUMN LAYOUT
# ==========================================
left_col, right_col = st.columns([2.2, 1])

# ==========================================
# LEFT SIDE (User Input)
# ==========================================
with left_col:
    st.subheader("📰 News Input")
    input_type = st.radio(
        "Choose Input",
        ["Paste News Text", "Upload News Image"]
    )

    news_text = ""

    if input_type == "Paste News Text":
        news_text = st.text_area(
            "",
            height=250,
            placeholder="Paste the complete news article here..."
        )
    else:
        uploaded_image = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg"]
        )
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )
            image_np = np.array(image)
            with st.spinner("Extracting text from image..."):
                result = ocr_reader.readtext(image_np, detail=0)
            news_text = " ".join(result)
            st.success("✅ Text Extracted Successfully")
            st.text_area(
                "Extracted Text",
                news_text,
                height=200
            )

    analyze = st.button(
        "🚀 Analyze News",
        use_container_width=True
    )

# ==========================================
# ANALYZE NEWS PROCESSING
# ==========================================
if analyze:
    if news_text.strip() == "":
        st.warning("⚠️ Please enter a news article.")
        st.stop()

    with st.spinner("🔍 TruthScope AI is analyzing the article..."):
        # 1. Vectorize input
        user_vector = vectorizer.transform([news_text])

        # 2. ML Prediction & Confidence
        prediction = model.predict(user_vector)[0]
        confidence = model.predict_proba(user_vector).max() * 100

        # 3. Find similar article via Cosine Similarity
        similarities = cosine_similarity(user_vector, dataset_vectors)
        best_index = similarities.argmax()
        matched_row = df.iloc[best_index]

        # 4. Retrieve values from Dataset match
        source_name = matched_row["source name"]
        fake_percentage = matched_row["fake news percentage"]
        credibility = matched_row["source credibility score"]
        sentiment = matched_row["sentiment"]

    # =====================================
    # DISPLAY RESULTS (Layout Split)
    # =====================================
    with left_col:
        st.markdown("---")
        if prediction == "Real":
            st.success("✅ Verified News")
        else:
            st.error("🚨 Fake News Detected")

        # Confidence
        st.markdown("##### 🎯 Prediction Confidence")
        st.write(f"**{confidence:.2f}%**")
        
        # Convert Fake % safely
        try:
            risk = float(fake_percentage)
            if risk <= 1:
                risk *= 100
            risk = round(risk)
            fake_percent = f"{risk}%"
        except:
            risk = 50
            fake_percent = str(fake_percentage)

        # =====================================
        # TRUST SCORE CALCULATION
        # =====================================

        try:
            credibility_score = float(str(credibility).split("/")[0])
        except:
            credibility_score = 5.0

        # Convert credibility from 10 -> 100
        credibility_percent = credibility_score * 10

        if prediction == "Real":
            trust_score = (
                confidence * 0.50 +
                credibility_percent * 0.30 +
                (100 - risk) * 0.20
            )
        else:
            trust_score = (
                (100 - confidence) * 0.50 +
                credibility_percent * 0.30 +
                (100 - risk) * 0.20
            )

        trust_score = round(max(0, min(100, trust_score))) 

        # =====================================
        # AI Verdict (Based on Trust Score)
        # =====================================

        if trust_score >= 85:
            ai_verdict = "🟢 VERIFIED"
        elif trust_score >= 60:
            ai_verdict = "🟠 NEEDS VERIFICATION"
        else:
            ai_verdict = "🔴 LIKELY FAKE" 

        # Detection Insights
        st.markdown("""
        <h4 style='font-size:22px; font-weight:600; margin-bottom:10px;'>
        📂 Detection Insights
        </h4>
        """, unsafe_allow_html=True)

        with st.expander("🏢 Source Name"):
            st.info(source_name)

        with st.expander("📊 Fake News Percentage"):
            st.warning(fake_percent)

        with st.expander("🛡️ Source Credibility Score"):
            st.success(credibility)

        with st.expander("🎭 Sentiment"):
            st.info(sentiment)
              
    # =====================================
    # AI GENERATION (Single Gemini API Call)
    # =====================================
    summary = ""
    explanation = ""

    if gemini_model is not None:
        if "ai_cache" not in st.session_state:
            st.session_state.ai_cache = {}

        # Check cache first
        if news_text in st.session_state.ai_cache:
            summary, explanation = st.session_state.ai_cache[news_text]
        else:
            combined_prompt = f"""
You are TruthScope AI.

Analyze the following news article.

Return your response exactly in this format.

### SUMMARY
Write exactly 3 bullet points.

### EXPLANATION
Explain in exactly 4 bullet points why the ML model predicted this result.

Prediction:
{prediction}

Confidence:
{confidence:.2f}%

News:
{news_text}
"""
            with st.spinner("Generating AI Insights..."):
                try:
                    response = gemini_model.generate_content(combined_prompt)
                    result = response.text

                    try:
                        summary = result.split("### SUMMARY")[1].split("### EXPLANATION")[0].strip()
                        explanation = result.split("### EXPLANATION")[1].strip()
                    except Exception:
                        summary = result
                        explanation = "Unable to extract explanation."

                    # Save to cache
                    st.session_state.ai_cache[news_text] = (summary, explanation)

                except Exception as e:
                    st.error(f"⚠️ {str(e)}")
            
        # =====================================
        # AI ANALYSIS DISPLAY
        # =====================================
        with left_col:
            st.markdown("""
            <h4 style='font-size:22px; font-weight:600; margin-bottom:10px;'>
            📰 AI Analysis
            </h4>
            """, unsafe_allow_html=True)

            with st.expander("📰 News Summary", expanded=False):
                st.info(summary)

            with st.expander("🧠 AI Explanation", expanded=False):
                st.success(explanation)
            
    # =====================================
    # SIDEBAR: AI INTELLIGENCE
    # =====================================
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; font-weight: 700;'>🤖 AI Intelligence</h2>", unsafe_allow_html=True)
        st.markdown("---")

        # 1. Option: Trust Score
        with st.expander("⭐ Trust Score", expanded=True):
            st.progress(trust_score / 100)
            st.markdown(f"<h3 style='text-align: center; color: #2563EB;'>{trust_score} / 100</h3>", unsafe_allow_html=True)

        # 2. Option: AI Verdict
        with st.expander("🤖 AI Verdict", expanded=False):
            if trust_score >= 85:
                st.success(f"Verdict: {ai_verdict}")
            elif trust_score >= 60:
                st.warning(f"Verdict: {ai_verdict}")
            else:
                st.error(f"Verdict: {ai_verdict}")

        # 3. Option: Fake News Risk
        with st.expander("📊 Fake News Risk", expanded=False):
            st.progress(risk / 100)
            st.markdown(f"<p style='font-size: 16px;'><b>Risk Level:</b> {fake_percent}</p>", unsafe_allow_html=True)

        # 4. Option: Analysis History
        with st.expander("🕒 Analysis History", expanded=False):
            if "history" not in st.session_state:
                st.session_state.history = []

            current_result = {
                "Prediction": prediction,
                "Trust": trust_score,
                "Risk": fake_percent
            }

            if current_result not in st.session_state.history:
                st.session_state.history.append(current_result)

            # Display history inside the option drawer
            if len(st.session_state.history) == 0:
                st.caption("No history available yet.")
            else:
                for item in reversed(st.session_state.history[-5:]):
                    color = "#EF4444" if item['Prediction'] == "Fake" else "#10B981"
                    st.markdown(
                        f"<div style='padding: 5px 0; border-bottom: 1px solid #e5e7eb;'>"
                        f"<span style='color: {color}; font-weight: bold;'>• {item['Prediction']}</span> | "
                        f"Trust: <b>{item['Trust']}/100</b> | Risk: <b>{item['Risk']}</b>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )