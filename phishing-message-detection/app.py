"""Streamlit demonstration for the trained model."""

from pathlib import Path

import streamlit as st

from phishdetect.inference import load_model, predict_message

MODEL_PATH = Path(__file__).parent / "artifacts" / "phishing_detector.joblib"

st.set_page_config(page_title="Phishing Message Detection", page_icon="🛡️")
st.title("Phishing Message Detection using NLP")
st.caption("Educational English-text classifier — never a substitute for security review.")

if not MODEL_PATH.exists():
    st.error("Model artifact missing. Run `python train.py` first.")
    st.stop()

model = load_model(MODEL_PATH)
message = st.text_area("Paste an email or message", height=180, placeholder="Subject and message body…")
if st.button("Analyze", type="primary"):
    try:
        result = predict_message(model, message)
    except ValueError as exc:
        st.warning(str(exc))
    else:
        st.subheader(result["label"].title())
        st.metric("Model phishing probability", f"{result['phishing_probability']:.1%}")
        st.caption(result["probability_note"])
        if result["influential_terms"]:
            st.write("Influential terms in this prediction")
            st.dataframe(result["influential_terms"], hide_index=True, use_container_width=True)
        st.info("Do not click links or open attachments based only on this demo. Verify through a trusted channel.")
