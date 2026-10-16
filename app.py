"""Optional web demo.  Run: streamlit run app.py   (needs: pip install streamlit)"""
import streamlit as st

from spamclf.predict import classify, load_model

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")
st.title("📩 SMS Spam Detector")


@st.cache_resource
def get_model():
    return load_model()


text = st.text_area("Paste an SMS message", height=140)
if st.button("Classify") and text.strip():
    label, score = classify(get_model(), [text])[0]
    (st.error if label == "SPAM" else st.success)(f"{label}  (spam score {score:.2f})")
