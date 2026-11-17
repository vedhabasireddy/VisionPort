import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch


# -----------------------------------------
# 1. Page Configuration
# -----------------------------------------

st.set_page_config(
    page_title="VisionPort",
    page_icon="🚗",
    layout="wide"
)


# -----------------------------------------
# 2. Title
# -----------------------------------------

st.title("🚦 VisionPort v0.1")

st.write(
    "GPU-Accelerated Traffic Object Detection"
)


# -----------------------------------------
# 3. Check GPU / CPU
# -----------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

st.write(f"Running on: **{device.upper()}**")


# -----------------------------------------
# 4. Load YOLO Model
# -----------------------------------------

@st.cache_resource
def load_model():

    model = YOLO("yolov8n.pt")

    return model


model = load_model()


# -----------------------------------------
# 5. Upload Image
# -----------------------------------------

uploaded_file = st.file_uploader(
    "Upload a traffic image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------------------
# 6. Process Image
# -----------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Original Image")

    st.image(
        image,
        caption="Uploaded Traffic Image",
        use_container_width=True
    )


    # -------------------------------------
    # 7. Run Object Detection
    # -------------------------------------

    results = model.predict(
        source=image,
        device=device
    )


    # -------------------------------------
    # 8. Get Detection Result
    # -------------------------------------

    result_image = results[0].plot()


    # -------------------------------------
    # 9. Display Detection Result
    # -------------------------------------

    st.subheader("Detection Result")

    st.image(
        result_image,
        caption="Detected Objects",
        use_container_width=True
    )