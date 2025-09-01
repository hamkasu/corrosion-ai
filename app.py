# app.py

import streamlit as st
import requests
import pandas as pd
import uuid
from PIL import Image
import os
from io import BytesIO

# === Set page config FIRST ===
st.set_page_config(
    page_title="Calmic Sdn Bhd | Corrosion Detection",
    page_icon="🔧",
    layout="wide"
)

# === Custom CSS for a beautiful UI ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&display=swap');
    body {
        font-family: 'Open Sans', sans-serif;
    }
    .header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }
    .logo {
        flex-shrink: 0;
    }
    .company-name {
        color: #2E8B57;
        font-size: 32px;
        font-weight: bold;
        margin: 0;
        padding: 0;
    }
    .tagline {
        color: #555555;
        font-size: 18px;
        margin: 0;
        padding: 0;
    }
    .stButton button {
        background-color: #2E8B57;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
    }
    .stButton button:hover {
        background-color: #236B45;
    }
    .stTextInput > label, .stTextArea > label {
        font-weight: 600;
    }
    hr {
        border: 1px solid #e0e0e0;
        margin: 20px 0;
    }
    .stApp {
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# === Header: Calmic Sdn Bhd ===
st.markdown('<div class="header">', unsafe_allow_html=True)

col_logo, col_title = st.columns([1, 4])

with col_logo:
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path)
            st.image(logo, width=100, output_format="PNG")
        except Exception as e:
            st.write("")
    else:
        st.write("")

with col_title:
    st.markdown('<h1 class="company-name">Calmic Sdn Bhd</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Corrosion Detection System</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Detect", "✍️ Manual Markup", "📊 Inspection Dashboard", "📦 Export Data"])

# --- TAB 1: Upload & Detect ---
with tab1:
    st.subheader("AI-Powered Corrosion Detection")

    col1, col2 = st.columns([1, 2])

    with col1:
        project_id = st.text_input("Project ID", value="default", help="e.g., Pipeline-2025")
        project_desc = st.text_area("Project Description")

    with col2:
        uploaded_file = st.file_uploader("Upload a metal surface image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("🔍 Analyzing with AI..."):
            try:
                filename = f"{uuid.uuid4()}_{uploaded_file.name}"
                image.save(f"uploads/{filename}")

                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"project_id": project_id, "project_description": project_desc}
                response = requests.post("http://localhost:8000/upload", files=files, data=data)
                results = response.json()

                pred = results["prediction"]
                conf = results["confidence"]

                if pred == "corrosion":
                    st.error(f"🔴 **CORROSION DETECTED** (Confidence: {conf:.1%})")
                else:
                    st.success(f"🟢 **NO CORROSION** (Confidence: {conf:.1%})")

                annotated_path = results["annotated_path"]
                st.image(f"http://localhost:8000{annotated_path}", caption="AI Detection Result", use_column_width=True)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# --- TAB 2: Manual Markup ---
with tab2:
    st.subheader("Manual Image Markup & Editing")

    if st.button("Load AI-Detected Image"):
        try:
            response = requests.get("http://localhost:8000/inspections/latest")
            latest = response.json()
            annotated_url = latest["annotated_path"]
            img_response = requests.get(f"http://localhost:8000{annotated_url}")
            img_data = img_response.content
            img = Image.open(BytesIO(img_data)).convert("RGB")
            st.session_state["draw_img"] = img.copy()
            st.session_state["annotations"] = []
        except Exception as e:
            st.error(f"Failed to load image: {e}")

    if "draw_img" in st.session_state:
        draw_img = st.session_state["draw_img"]

        color = st.color_picker("Line color", "#FF0000")
        stroke_width = st.slider("Line thickness", 1, 10, 3)
        annotation_text = st.text_input("Add note", placeholder="e.g., Surface pitting near weld")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(draw_img, caption="Marked-up Image", use_column_width=True)
        with col2:
            x1 = st.number_input("X1", 0, draw_img.width, 50)
            y1 = st.number_input("Y1", 0, draw_img.height, 50)
            x2 = st.number_input("X2", 0, draw_img.width, 150)
            y2 = st.number_input("Y2", 0, draw_img.height, 150)

        if st.button("Add Box"):
            draw = ImageDraw.Draw(draw_img)
            rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            draw.rectangle([x1, y1, x2, y2], outline=rgb, width=stroke_width)
            if annotation_text:
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                draw.text((x1, y1-25), annotation_text, fill=rgb, font=font)
            st.session_state["draw_img"] = draw_img

        buf = BytesIO()
        draw_img.save(buf, format="JPEG")
        byte_img = buf.getvalue()

        st.download_button(
            label="💾 Save Marked-up Image",
            data=byte_img,
            file_name=f"marked_{uuid.uuid4()}.jpg",
            mime="image/jpeg"
        )

# --- TAB 3: View Data ---
with tab3:
    st.subheader("📋 Inspection Records")
    try:
        response = requests.get("http://localhost:8000/inspections")
        if response.status_code == 200:
            inspections = response.json()
            df = pd.DataFrame(inspections)
            df = df[["id", "project_id", "prediction", "confidence", "uploaded_at"]]
            df["uploaded_at"] = pd.to_datetime(df["uploaded_at"]).dt.strftime("%Y-%m-%d %H:%M")
            df["confidence"] = (df["confidence"] * 100).round(1).astype(str) + "%"
            st.dataframe(df, use_container_width=True)
        else:
            st.error("Failed to load inspection data.")
    except Exception as e:
        st.error(f"Connection error: {e}")

# --- TAB 4: Export ---
with tab4:
    st.subheader("Export Inspection Data")
    if st.button("📥 Download CSV"):
        try:
            url = "http://localhost:8000/inspections/export"
            response = requests.get(url)
            if response.status_code == 200:
                st.download_button(
                    label="✅ Click to download CSV",
                    data=response.content,
                    file_name="calmic_corrosion_inspections.csv",
                    mime="text/csv"
                )
            else:
                st.error("Export failed.")
        except Exception as e:
            st.error(f"Error: {e}")

# === Footer ===
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #777; font-size: 14px;'>"
    "© 2025 Calmic Sdn Bhd | Corrosion Detection AI System | All data is securely stored in the cloud</p>",
    unsafe_allow_html=True
)