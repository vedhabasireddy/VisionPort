
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import pandas as pd

# -----------------------------------------
# 1. Page Configuration
# -----------------------------------------

st.set_page_config(
    page_title="VisionPort",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------
# 2. Load Model
# -----------------------------------------

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -----------------------------------------
# 3. Device Selection
# -----------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------
# 4. Sidebar
# -----------------------------------------

st.sidebar.title("VisionPort Controls")

st.sidebar.write("Upload a traffic image")

uploaded_file = st.sidebar.file_uploader(
    "Choose IDD image",
    type=["jpg", "jpeg", "png"]
)

st.sidebar.divider()

conf = st.sidebar.slider(
    "Confidence Threshold",
    0.10, 0.90, 0.25, 0.05
)

iou = st.sidebar.slider(
    "IoU Threshold",
    0.10, 0.90, 0.50, 0.05
)

img_size = st.sidebar.selectbox(
    "Image Size",
    [640, 800, 960, 1280]
)

st.sidebar.divider()

st.sidebar.write(f"Execution Device: **{device.upper()}**")

if device == "cuda":
    st.sidebar.success("GPU available")
else:
    st.sidebar.info("Running on CPU")

# -----------------------------------------
# 5. Main Page
# -----------------------------------------

st.title("🚦 VisionPort v0.1")

st.write(
    "GPU-Accelerated Indian Traffic Object Detection"
)

st.divider()

# -----------------------------------------
# 6. Image Processing
# -----------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Detecting objects..."):

        results = model.predict(
            source=image,
            device=device,
            conf=conf,
            iou=iou,
            imgsz=img_size,
            verbose=False
        )

    result = results[0]

    # Convert annotated image from BGR to RGB
    result_image = result.plot()
    result_image = result_image[:, :, ::-1]

    # -------------------------------------
    # 7. Display Images
    # -------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detection Result")
        st.image(result_image, use_container_width=True)

    st.divider()

    # -------------------------------------
    # 8. Detection Summary
    # -------------------------------------

    st.subheader("Detection Summary")

    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:

        class_ids = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()

        names = result.names

        detected_objects = {}

        for class_id, score in zip(class_ids, scores):

            name = names[class_id]

            if name not in detected_objects:
                detected_objects[name] = {
                    "Count": 0,
                    "Scores": []
                }

            detected_objects[name]["Count"] += 1
            detected_objects[name]["Scores"].append(
                float(score)
            )

        total = len(boxes)

        st.metric("Total Objects Detected", total)

        summary = []

        for name, details in detected_objects.items():

            summary.append({
                "Object": name,
                "Count": details["Count"],
                "Average Confidence": round(
                    sum(details["Scores"]) /
                    len(details["Scores"]), 3
                )
            })

        st.dataframe(
            pd.DataFrame(summary),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("No objects detected.")

else:

    st.info(
        "Upload an IDD traffic image using the sidebar "
        "to begin object detection."
    )
