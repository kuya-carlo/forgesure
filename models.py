"""
Pydantic models for FastAPI request/response serialization.

This module defines all data structures used in the API for type validation
and documentation generation.
"""

from typing import List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field


class PlateRequest(BaseModel):
    """Represents a steel plate with manufacturing specifications."""
    
    id: str = Field(..., default_factory=uuid4,description="Unique plate identifier")
    thickness: float = Field(..., description="Plate thickness in mm")
    length: float = Field(..., description="Plate length in mm")
    width: float = Field(..., description="Plate width in mm")
    holes: int = Field(..., description="Number of holes to drill")
    beam_size: str = Field(..., description="Beam dimensions (e.g., '20x12')")
    price: float = Field(..., description="Selling price in PHP")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "P001",
                "thickness": 12,
                "length": 8000,
                "width": 2000,
                "holes": 4,
                "beam_size": "20x12",
                "price": 5000
            }
        }


class OptimizationConstraints(BaseModel):
    """Constraints for the optimization problem."""
    
    max_shift_hours: float = Field(
        default=8.0,
        description="Maximum shift duration in hours"
    )
    max_plates_per_shift: int = Field(
        default=3,
        description="Maximum number of plates per shift"
    )
    material_cost_per_ton: float = Field(
        default=500.0,
        description="Material cost per ton in PHP"
    )
    energy_cost_per_kwh: float = Field(
        default=5.0,
        description="Energy cost per kWh in PHP"
    )
    labor_cost_per_hour: float = Field(
        default=50.0,
        description="Labor cost per hour in PHP"
    )


class OptimizationWeights(BaseModel):
    """Weights for multi-objective optimization."""
    
    profit_weight: float = Field(
        default=1.0,
        description="Weight for profit maximization"
    )
    nesting_weight: float = Field(
        default=0.5,
        description="Weight for nesting utilization"
    )
    first_pass_weight: float = Field(
        default=0.5,
        description="Weight for first-pass yield"
    )
    energy_weight: float = Field(
        default=-0.3,
        description="Weight for energy intensity (negative to minimize)"
    )


class OptimizationRequest(BaseModel):
    """Request body for plate optimization."""
    
    plates: List[PlateRequest] = Field(..., description="List of plates to optimize")
    constraints: Optional[OptimizationConstraints] = Field(
        default_factory=OptimizationConstraints,
        description="Optimization constraints"
    )
    weights: Optional[OptimizationWeights] = Field(
        default_factory=OptimizationWeights,
        description="Objective function weights"
    )


class PlateKPIResponse(BaseModel):
    """Key performance indicators for a scheduled plate."""
    
    id: str
    profit: float = Field(description="Estimated profit in PHP")
    nesting_utilization: float = Field(description="Material utilization percentage")
    first_pass_yield: float = Field(description="Production quality indicator")
    energy_intensity_kwh_per_ton: float = Field(description="Energy consumption per ton")
    material_cost: float
    energy_cost: float
    labor_cost: float
    total_process_time: float = Field(description="Total manufacturing time in hours")


class OptimizationResponse(BaseModel):
    """Response containing optimization results."""
    
    status: str = Field(description="Optimization status (success/failed)")
    scheduled_plates: List[str] = Field(description="IDs of scheduled plates")
    kpis: List[PlateKPIResponse] = Field(description="KPIs for scheduled plates")
    total_profit: float = Field(description="Total estimated profit")
    total_process_time: float = Field(description="Total shift time used")
    optimization_value: float = Field(
        description="Objective function value (higher is better)"
    )
