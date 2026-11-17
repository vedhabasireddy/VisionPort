# VisionPort

YOLOv8-based object detection project with a Streamlit UI and a FastAPI gateway.

## Files
- `main.py` – Streamlit app (VisionPort UI)
- `app.py` – additional app script
- `gateway.py` / `gateway_trace.py` – FastAPI gateway services
- `yolov8n.pt` – YOLOv8 nano weights

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Run
```bash
streamlit run main.py
uvicorn gateway:app --reload
```
