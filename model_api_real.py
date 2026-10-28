from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO

import cv2
import numpy as np
import torch
import uvicorn


app = FastAPI()

MODEL_PATH = "best.pt"

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("Loading model...")
print("Device:", device)

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")


def make_rgbt_tensor(rgb_bytes, thermal_bytes, imgsz=640):

    rgb_array = np.frombuffer(rgb_bytes, np.uint8)
    thermal_array = np.frombuffer(thermal_bytes, np.uint8)

    rgb_img = cv2.imdecode(
        rgb_array,
        cv2.IMREAD_COLOR
    )

    thermal_img = cv2.imdecode(
        thermal_array,
        cv2.IMREAD_UNCHANGED
    )

    if rgb_img is None:
        raise ValueError("RGB image cannot be read")

    if thermal_img is None:
        raise ValueError("Thermal image cannot be read")

    if thermal_img.ndim == 3:
        thermal_img = cv2.cvtColor(
            thermal_img,
            cv2.COLOR_BGR2GRAY
        )

    rgb_img = cv2.resize(
        rgb_img,
        (imgsz, imgsz)
    )

    thermal_img = cv2.resize(
        thermal_img,
        (imgsz, imgsz)
    )

    rgb_img = cv2.cvtColor(
        rgb_img,
        cv2.COLOR_BGR2RGB
    )

    rgb_img = rgb_img.astype(
        np.float32
    ) / 255.0

    thermal_img = thermal_img.astype(
        np.float32
    ) / 255.0

    thermal_img = thermal_img[..., None]

    rgbt = np.concatenate(
        [rgb_img, thermal_img],
        axis=2
    )

    tensor = torch.from_numpy(
        rgbt
    ).permute(2, 0, 1)

    tensor = tensor.unsqueeze(0)

    return tensor.to(device)


@app.get("/")
def home():
    return {
        "message": "RGB-T YOLO API is running",
        "device": str(device)
    }


@app.post("/predict")
async def predict(
    rgb: UploadFile = File(...),
    thermal: UploadFile = File(...)
):

    rgb_bytes = await rgb.read()
    thermal_bytes = await thermal.read()

    im = make_rgbt_tensor(
        rgb_bytes,
        thermal_bytes,
        imgsz=640
    )

    with torch.no_grad():

        prediction = model.predict(
            im,
            verbose=False,
            channels=4,
            use_simotm="RGBT",
            conf=0.1
        )[0]

    detections = []

    for box in prediction.boxes:

        class_id = int(
            box.cls[0].cpu().item()
        )

        confidence = float(
            box.conf[0].cpu().item()
        )

        bbox = (
            box.xyxy[0]
            .cpu()
            .tolist()
        )

        if class_id == 0:
            class_name = "real_person"

        elif class_id == 1:
            class_name = "mannequin"

        else:
            class_name = f"class_{class_id}"

        detections.append({
            "class_id": class_id,
            "class": class_name,
            "confidence": round(confidence, 4),
            "bbox": [
                round(x, 2)
                for x in bbox
            ]
        })

    real_count = sum(
        1
        for d in detections
        if d["class_id"] == 0
    )

    mannequin_count = sum(
        1
        for d in detections
        if d["class_id"] == 1
    )

    return {
        "status": "success",

        "rgb_filename": rgb.filename,
        "thermal_filename": thermal.filename,

        "summary": {
            "total": len(detections),
            "real_person": real_count,
            "mannequin": mannequin_count
        },

        "detections": detections
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
