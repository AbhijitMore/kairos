from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class AdultInstance(BaseModel):
    age: int = Field(..., json_schema_extra={"examples": [39]})
    workclass: str = Field(..., json_schema_extra={"examples": ["State-gov"]})
    education_num: int = Field(..., json_schema_extra={"examples": [13]})
    marital_status: str = Field(..., json_schema_extra={"examples": ["Never-married"]})
    occupation: str = Field(..., json_schema_extra={"examples": ["Adm-clerical"]})
    relationship: str = Field(..., json_schema_extra={"examples": ["Not-in-family"]})
    race: str = Field(..., json_schema_extra={"examples": ["White"]})
    sex: str = Field(..., json_schema_extra={"examples": ["Male"]})
    capital_gain: int = Field(..., json_schema_extra={"examples": [2174]})
    capital_loss: int = Field(..., json_schema_extra={"examples": [0]})
    hours_per_week: int = Field(..., json_schema_extra={"examples": [40]})
    native_country: str = Field(..., json_schema_extra={"examples": ["United-States"]})

class BatchInferenceRequest(BaseModel):
    instances: List[AdultInstance]

class PredictionResponse(BaseModel):
    decision: Literal["ACCEPT", "REJECT", "ABSTAIN"]
    probability: float
    uncertainty: float
    cost_risk: float
