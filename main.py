from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import shutil
import os
from ultralytics import YOLO

app = FastAPI()

model = YOLO('yolov8n.pt')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_extension = file.filename.split('.')[-1].lower()
    valid_extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "tif", "gif"]
    
    if file_extension not in valid_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    
    input_path = f"input.{file_extension}"
    
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save input file: {str(e)}")
    
    try:
        results = model(input_path)
        
        output_image_path = results[0].save()
        
        os.remove(input_path)
        
        if os.path.exists(output_image_path):
            return FileResponse(output_image_path)
        else:
            raise HTTPException(status_code=500, detail="Output image not found after inference.")
    
    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")