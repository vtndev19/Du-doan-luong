import pandas as pd
import joblib
import os
import glob
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

# ======================================================================
# 1. CÁC HÀM TIỀN XỬ LÝ ĐỂ KHỚP VỚI DỮ LIỆU ĐẦU VÀO CỦA MODEL
# ======================================================================

def convert_to_string(df):
    return df.astype(str)

def preprocess_input(df):
    df_processed = df.copy()

    df_processed['description'] = df_processed.get('description', "").fillna("")
    df_processed['skills_desc'] = df_processed.get('skills_desc', "").fillna("")

    df_processed["all_text"] = (
        df_processed["description"] + " " + df_processed["skills_desc"]
    )

    return df_processed

# ======================================================================
# 2. SCHEMA CHO 1 JOB
# ======================================================================

class JobPosting(BaseModel):
    company_name: Optional[str] = "Unknown"
    title: str
    description: Optional[str] = ""
    location: Optional[str] = "Remote"
    views: Optional[float] = 0.0
    formatted_work_type: Optional[str] = "Full-time"
    remote_allowed: Optional[str] = "0"
    formatted_experience_level: Optional[str] = "Entry Level"
    skills_desc: Optional[str] = ""
    sponsored: Optional[int] = 0
    application_type: Optional[str] = "Simple"

class BatchRequest(BaseModel):
    data: List[JobPosting]

# ======================================================================
# 3. INIT FASTAPI + CORS
# ======================================================================

app = FastAPI(
    title="Salary Prediction API",
    description="Predict salary from job posting using ML model",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_pipeline = None

# ======================================================================
# 4. LOAD MODEL
# ======================================================================

@app.on_event("startup")
def load_model():
    global model_pipeline

    print("Đang tìm model trong thư mục /models...")

    model_dir = "models"
    files = glob.glob(os.path.join(model_dir, "lightgbm_tuned_model_*.pkl"))

    if not files:
        print("KHÔNG TÌM THẤY MODEL!")
        return

    latest_model = max(files, key=os.path.getctime)

    print(f"Loading model: {latest_model}")

    try:
        model_pipeline = joblib.load(latest_model)
        print("Model đã sẵn sàng để dự đoán!")
    except Exception as e:
        print("Lỗi khi load model:", e)

# ======================================================================
# 5. TEST API
# ======================================================================

@app.get("/")
def home():
    return {"message": "ML Salary Prediction API Running"}

# ======================================================================
# 6. API DỰ ĐOÁN 1 JOB
# ======================================================================

@app.post("/predict")
def predict(job: JobPosting):
    global model_pipeline

    print("\n=== NHẬN REQUEST /predict ===")
    print("DATA NHẬN ĐƯỢC:", job.dict())

    if model_pipeline is None:
        raise HTTPException(500, "Model chưa được load")

    df = pd.DataFrame([job.dict()])
    df_ready = preprocess_input(df)

    pred = float(model_pipeline.predict(df_ready)[0])

    print(f"DỰ ĐOÁN HOÀN TẤT: {pred}\n")

    return {
        "title": job.title,
        "predicted_salary": pred
    }

# ======================================================================
# 7. API DỰ ĐOÁN NHIỀU JOB
# ======================================================================

@app.post("/predict/batch")
def predict_batch(batch: BatchRequest):
    global model_pipeline

    print("\n=== NHẬN REQUEST /predict/batch ===")
    print(f"SỐ JOB NHẬN ĐƯỢC: {len(batch.data)}")

    # In từng job (chỉ tên + 20 ký tự đầu mô tả)
    for idx, job in enumerate(batch.data):
        print(f"Job #{idx+1}: {job.title} | desc: {job.description[:30]}...")

    if model_pipeline is None:
        raise HTTPException(500, "Model chưa được load")

    jobs_list = [job.dict() for job in batch.data]
    df = pd.DataFrame(jobs_list)

    df_ready = preprocess_input(df)
    preds = model_pipeline.predict(df_ready)

    results = []
    for job, salary in zip(batch.data, preds):
        results.append({
            "title": job.title,
            "predicted_salary": float(salary),
        })

    print("DỰ ĐOÁN HOÀN TẤT CHO BATCH\n")

    return {"predictions": results}

# ======================================================================
# 8. RUN SERVER
# ======================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
