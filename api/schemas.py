# api/schemas.py
from pydantic import BaseModel
from typing import List, Dict, Optional

class PredictionOut(BaseModel):
    performance_score: float
    injury_risk: int
    top_features: List[Dict[str, float]]
