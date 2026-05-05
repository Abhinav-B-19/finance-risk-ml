from pydantic import BaseModel

class PredictionInput(BaseModel):
    income: float
    expenses: float
    debt: float
    shock: str = "none"