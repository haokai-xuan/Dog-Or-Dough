from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np

import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

API_TOKEN = os.getenv("API_TOKEN")

# Load custom NN
from model import NeuralNetwork
NN = NeuralNetwork(load_path=str(BASE_DIR / "model_parameters.npz"))

# Load pytorch-trained onnx model
import onnxruntime as ort
session = ort.InferenceSession(BASE_DIR / "model.onnx", providers=["CPUExecutionProvider"])
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
input_name = session.get_inputs()[0].name
ort_inputs = {input_name: input_data}

# Preprocess img for onnx inferencing
def onnx_preprocess(img):
    img = img.resize((232, 232), Image.Resampling.BILINEAR)
    img = np.array(img.convert("RGB"))
    img = img.astype(np.float32) / 255.0

    # Center crop
    start_x, start_y = 4, 4
    end_x, end_y = 228, 228
    img = img[start_y:end_y, start_x:end_x]

    img = img.transpose(2, 0, 1) # (C, H, W)

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    img = np.expand_dims(img, axis=0)
    return img

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://dog-or-dough.vercel.app"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...), api_key: str | None = Header(alias="api-key"), model_type: str = Query(default="Linear")):
    if api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    if model_type == "Linear":
        image = image.convert("RGB")
        image = image.resize((64, 64), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0 # (64, 64, 3) RGB
        arr = arr[:, :, ::-1] # Network learned in BGR
        arr = arr.transpose(2, 0, 1) # (H, W, C) -> (C, H, W)
        arr = arr.reshape(-1, 1)

        _, probs = NN.predict(arr)
        probs = probs.flatten()
    else:
        input_tensor = onnx_preprocess(image)
        outputs = session.run(None, {input_name: input_tensor})[0]
        
        shifted = outputs - np.max(outputs, axis=1, keepdims=True)
        probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
        probs = probs.flatten()

    return JSONResponse({
        "dog": float(probs[0]),
        "dough": float(probs[1])
    })