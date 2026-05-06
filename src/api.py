from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.predict import predict

# ─────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────
app = FastAPI(
    title="Finance Risk ML API",
    description="AI-based financial distress forecasting service",
    version="1.0.0"
)

# ─────────────────────────────────────────
# CORS
# ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# REQUEST MODEL
# ─────────────────────────────────────────
class PredictionInput(BaseModel):
    income: float
    expenses: float
    debt: float

    shock: str = "none"

    dti_lag1: float = 0
    dti_lag2: float = 0

    savings_ratio_lag1: float = 0
    savings_ratio_lag2: float = 0

    debt_lag1: float = 0
    debt_lag2: float = 0


# ─────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Finance Risk ML API Running"
    }


# ─────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "finance-risk-ml"
    }


# ─────────────────────────────────────────
# PREDICTION ENDPOINT
# ─────────────────────────────────────────
@app.post("/predict")
def predict_risk(data: PredictionInput):

    result = predict(data.dict(), explain=False)

    return result


# ─────────────────────────────────────────
# GLOBAL ERROR HANDLER
# ─────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": str(exc)
        }
    )