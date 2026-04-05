from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openpyxl import Workbook, load_workbook
import os
from datetime import datetime

from  main import final,extract_article,logs_collection,save_to_mongodb
app=FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsInput(BaseModel):
    title: str
    text: str
class URLInput(BaseModel):
    url: str

@app.get("/")
def home():
    return FileResponse("static/index.html")
@app.post("/predict")
def predict(data: NewsInput, request: Request):
    ip = request.client.host
    result = final(data.title, data.text)
    save_to_mongodb({
        "type": "TEXT",
        "input": f"{data.title} (IP: {ip})",
        "prediction": result["model_prediction"],
        "confidence": result["confidence"],
        "verification": result["fact_check_status"],
        "matched": result["matched_sources"],
        "ai_verdict": result["Ai_verdict"],
        "ai_reason": result["Ai_reason"]
    })
    
    return result

@app.post("/predict-url")
def predict_from_url(data: URLInput):
    try:
        article = extract_article(data.url)
        result = final(article["title"], article["text"], data.url)
        save_to_mongodb({
            "type": "URL",
            "input": data.url,
            "prediction": result["model_prediction"],
            "confidence": result["confidence"],
            "verification": result["fact_check_status"],
            "matched": result["matched_sources"],
            "ai_verdict": result["Ai_verdict"],
            "ai_reason": result["Ai_reason"]
        })
        
        return result
    
    except Exception as e:
        return {"error": str(e)}
@app.get("/logs")
def get_logs():
    try:
        raw_logs = list(logs_collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(50))
        
        return {"logs": raw_logs}
    
    except Exception as e:
        return {"error": f"Could not fetch logs from MongoDB: {str(e)}"}      