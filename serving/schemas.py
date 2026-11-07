"""Request/response contracts. Validation ranges mirror data_pipeline.VALID_RANGES so the API rejects
exactly what the training pipeline would have dropped, instead of silently scoring garbage."""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

EmploymentType = Literal["contract", "government", "salaried", "self_employed"]
LoanPurpose = Literal["business", "education", "home", "medical", "personal", "vehicle"]


class Application(BaseModel):
    model_config = ConfigDict(extra="forbid")          # a typo'd field name is an error, not a silent default
    application_id: Optional[str] = Field(None, max_length=64)
    age: int = Field(..., ge=18, le=100)
    income_annual_inr: float = Field(..., gt=0)
    employment_type: EmploymentType
    employment_years: float = Field(..., ge=0, le=60)
    existing_loans: int = Field(..., ge=0, le=50)
    existing_emi_inr: float = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=900)
    loan_amount_inr: float = Field(..., gt=0)
    loan_purpose: LoanPurpose
    ltv_ratio: float = Field(..., ge=0, le=2.0)
    dti_ratio: Optional[float] = Field(None, ge=0, description="Derived from EMI, income and loan amount if omitted")

    @model_validator(mode="after")
    def _derive_dti(self):
        if self.dti_ratio is None:
            # In the provided data: dti = EMI*12/income + 0.096*loan/income (R^2 = 0.99997)
            self.dti_ratio = round(self.existing_emi_inr * 12 / self.income_annual_inr
                                   + 0.096 * self.loan_amount_inr / self.income_annual_inr, 2)
        return self


class Factor(BaseModel):
    feature: str
    label: str
    value: str
    shap_logodds: float


class WaterfallStep(BaseModel):
    label: str
    shap_logodds: float
    start: float
    end: float


class Waterfall(BaseModel):
    base_value: float
    final_value: float
    steps: List[WaterfallStep]


class Prediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())   # field `model_version` is intentional
    application_id: Optional[str] = None
    default_probability: float = Field(..., description="Raw model output, NOT calibrated (trained with scale_pos_weight)")
    score: int = Field(..., description="300-900, higher is safer: 300 + 600*(1-p)")
    risk_band: Literal["low", "medium", "high"]
    top_risk_factors: List[Factor]
    top_protective_factors: List[Factor]
    waterfall: Waterfall
    model_version: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    applications: List[Application] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    count: int
    predictions: List[Prediction]
