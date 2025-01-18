from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
from ultralytics import YOLO

app = FastAPI()

model = YOLO('yolov8n.pt')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_extension = file.filename.split('.')[-1].lower()
    image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "tif"]
    video_extensions = ["mp4", "avi", "mov", "mkv", "gif", "wmv"]
    
    if file_extension not in image_extensions + video_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    input_path = f"input.{file_extension}"
    
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save input file: {str(e)}")
    
    try:
        results = model(input_path, save=True, project=".//", name="prediction", exist_ok=True)
        
        os.remove(input_path)
        
        if file_extension in image_extensions:
            if os.path.exists(".//prediction//input.jpg"):
                return FileResponse(".//prediction//input.jpg")
            else:
                raise HTTPException(status_code=500, detail="Output image not found after inference")
        else:
            if os.path.exists(".//prediction//input.avi"):
                return FileResponse(".//prediction//input.avi")
            else:
                raise HTTPException(status_code=500, detail="Output image not found after inference")
    
    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")