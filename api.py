from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from pathlib import Path

app = FastAPI(title="HelmetGuard AI API", version="1.0")

MODEL_PATH = Path(__file__).parent / "model" / "best.pt"
model = YOLO(str(MODEL_PATH))

@app.get("/")
def home():
    return {"project": "HelmetGuard AI", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image")

    result = model.predict(image, imgsz=640, conf=0.10, verbose=False)[0]
    detections = []

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        class_name = model.names[cls_id]

        threshold = 0.30 if class_name == "helmet" else 0.45
        if confidence < threshold:
            continue

        detections.append({
            "class": class_name,
            "confidence": round(confidence, 4),
            "bbox_xyxy": [round(float(v), 2) for v in box.xyxy[0].tolist()]
        })

    return {
        "filename": file.filename,
        "detections_count": len(detections),
        "detections": detections
    }
