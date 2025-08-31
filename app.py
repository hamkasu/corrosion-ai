# app.py

import streamlit as st
from PIL import Image

# Import your model
from app.model import predict_with_boxes

# Page config
st.set_page_config(page_title="🔧 Corrosion Detection", page_icon="🔧", layout="centered")

# Title
st.title("Corrosion Detection System")
st.markdown("Upload an image to detect corrosion")

# File uploader
uploaded_file = st.file_uploader("Upload a metal surface image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Analyzing for corrosion..."):
        try:
            results = predict_with_boxes(image)
            label = results["label"]
            confidence = results["confidence"]
            annotated_image = results["annotated_image"]

            if label == "corrosion":
                st.success(f"✅ Prediction: **CORROSION DETECTED**")
            else:
                st.warning(f"❌ Prediction: **NO CORROSION**")

            st.metric("Confidence", f"{confidence:.1%}")
            st.image(annotated_image, caption="Detected Corrosion Areas (Dashed Boxes)", use_column_width=True)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Powered by YOLOv8 and Streamlit*")