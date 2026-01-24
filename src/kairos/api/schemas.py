from pydantic import BaseModel, Field
from typing import List, Literal, Union, Annotated


class AdultInstance(BaseModel):
    dataset_type: Literal["adult"] = "adult"
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


class HomeCreditInstance(BaseModel):
    dataset_type: Literal["home_credit"] = "home_credit"
    AMT_INCOME_TOTAL: float = Field(..., json_schema_extra={"examples": [202500.0]})
    AMT_CREDIT: float = Field(..., json_schema_extra={"examples": [406597.5]})
    AMT_ANNUITY: float = Field(..., json_schema_extra={"examples": [24700.5]})
    AMT_GOODS_PRICE: float = Field(..., json_schema_extra={"examples": [351000.0]})
    REGION_RATING_CLIENT: int = Field(..., json_schema_extra={"examples": [2]})
    DAYS_BIRTH: int = Field(..., json_schema_extra={"examples": [-9461]})
    DAYS_EMPLOYED: int = Field(..., json_schema_extra={"examples": [-637]})
    EXT_SOURCE_1: float = Field(..., json_schema_extra={"examples": [0.083]})
    EXT_SOURCE_2: float = Field(..., json_schema_extra={"examples": [0.262]})
    EXT_SOURCE_3: float = Field(..., json_schema_extra={"examples": [0.139]})
    NAME_EDUCATION_TYPE: str = Field(..., json_schema_extra={"examples": ["Secondary"]})
    NAME_INCOME_TYPE: str = Field(..., json_schema_extra={"examples": ["Working"]})
    OCCUPATION_TYPE: str = Field(..., json_schema_extra={"examples": ["Laborers"]})


class BatchInferenceRequest(BaseModel):
    dataset: Literal["adult", "home_credit"] = "adult"
    instances: List[
        Annotated[
            Union[AdultInstance, HomeCreditInstance],
            Field(discriminator="dataset_type"),
        ]
    ]


class PredictionResponse(BaseModel):
    decision: Literal["ACCEPT", "REJECT", "ABSTAIN"]
    probability: float
    uncertainty: float
    cost_risk: float
