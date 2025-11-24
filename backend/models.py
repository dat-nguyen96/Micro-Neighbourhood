# backend/models.py - Pydantic Models for API
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class NeighbourhoodMetrics(BaseModel):
    bevolkingsdichtheid: Optional[float] = Field(None, description="Bevolkingsdichtheid per km²")
    gemiddelde_huishoudgrootte: Optional[float] = Field(None, description="Gemiddeld aantal personen per huishouden")
    percentage_kinderen: Optional[float] = Field(None, description="Percentage kinderen (0-15 jaar)")
    percentage_jongeren: Optional[float] = Field(None, description="Percentage jongeren (15-25 jaar)")
    percentage_ouderen: Optional[float] = Field(None, description="Percentage ouderen (65+ jaar)")
    stedelijkheid_score: Optional[float] = Field(None, description="Stedelijkheid score (1-5)")
    geweld_per_1000: Optional[float] = Field(None, description="Geweldsincidenten per 1000 inwoners")


class NeighbourhoodStoryRequest(BaseModel):
    buurt_code: str = Field(..., description="CBS buurt code (bijv. BU03630001)")
    metrics: NeighbourhoodMetrics
    gemeente: str = Field(..., description="Gemeente naam")
    buurt_naam: str = Field(..., description="Buurt naam")


class NeighbourhoodStoryResponse(BaseModel):
    story: str = Field(..., description="Gegenereerde verhaal over de buurt")


class SimilarBuurt(BaseModel):
    buurt_code: str
    naam: Optional[str] = None
    gemeente: str
    distance: float
    cluster: int
    cluster_label_short: str
    population: Optional[float] = None
    income_per_person: Optional[float] = None
    pca_x: Optional[float] = None
    pca_y: Optional[float] = None


class SimilarBuurtenResponse(BaseModel):
    base_buurt_code: str
    base_cluster_label_short: Optional[str] = None
    base_cluster_label_long: Optional[str] = None
    base_pca_x: Optional[float] = None
    base_pca_y: Optional[float] = None
    neighbours: List[SimilarBuurt]


class ClusterInfoResponse(BaseModel):
    buurt_code: str
    buurt_naam: str     # echte buurt naam
    gemeente_naam: str  # gemeente naam
    cluster: int
    label: str          # korte label (voor badge)
    label_long: str     # lange uitleg (voor panel)
