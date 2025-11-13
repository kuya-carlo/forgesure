"""
FastAPI application for steel plate production optimization.

This module implements REST API endpoints for:
- Health check (ping)
- Production optimization using linear programming

The API uses OOP principles with service classes handling business logic
and Pydantic models for request/response validation.

Example usage:
    curl http://localhost:8000/ping
    curl -X POST http://localhost:8000/optimize -H "Content-Type: application/json" \
        -d @request.json
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from plate_service import PlateOptimizationService
from models import (
    OptimizationRequest,
    OptimizationResponse,
    OptimizationConstraints,
    OptimizationWeights
)

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Steel Plate Production Optimizer API",
    description="REST API for optimizing steel plate production scheduling using linear programming",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize the optimization service
optimization_service = PlateOptimizationService()


@app.get(
    "/ping",
    tags=["Health"],
    summary="Health check endpoint",
    description="Simple endpoint to verify API is running"
)
async def ping() -> JSONResponse:
    """
    Health check endpoint.
    
    Returns:
        JSON response with status and message
        
    Example:
        GET /ping
        Response: {"status": "ok", "message": "API is running"}
    """
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "message": "API is running"}
    )


@app.post(
    "/optimize",
    response_model=OptimizationResponse,
    tags=["Optimization"],
    summary="Optimize plate production schedule",
    description="Solves a linear programming problem to maximize profit while considering "
                "nesting utilization, first-pass yield, and energy intensity. "
                "Constraints: shift time and max plates per shift."
)
async def optimize_production(request: OptimizationRequest) -> OptimizationResponse:
    """
    Optimize plate production schedule.
    
    This endpoint takes a list of available steel plates and optimization parameters,
    then solves a multi-objective linear programming problem to determine which plates
    should be manufactured in the next shift.
    
    The optimization considers:
    - Profit maximization (selling price - costs)
    - Material utilization (nesting efficiency)
    - Production quality (first-pass yield)
    - Energy efficiency (energy intensity per ton)
    
    Args:
        request: OptimizationRequest containing:
            - plates: List of plates with specifications
            - constraints: Production constraints (optional, uses defaults if not provided)
            - weights: Objective function weights (optional, uses defaults if not provided)
    
    Returns:
        OptimizationResponse containing:
            - status: "success" or "failed"
            - scheduled_plates: IDs of selected plates
            - kpis: Performance metrics for each scheduled plate
            - total_profit: Sum of profits from scheduled plates
            - total_process_time: Total manufacturing time
            - optimization_value: Objective function value
    
    Raises:
        HTTPException: If optimization fails or invalid input provided
        
    Example request:
        {
            "plates": [
                {
                    "id": "P001",
                    "thickness": 12,
                    "length": 8000,
                    "width": 2000,
                    "holes": 4,
                    "beam_size": "20x12",
                    "price": 5000
                }
            ],
            "constraints": {
                "max_shift_hours": 8.0,
                "max_plates_per_shift": 3,
                "material_cost_per_ton": 500,
                "energy_cost_per_kwh": 5,
                "labor_cost_per_hour": 50
            },
            "weights": {
                "profit_weight": 1.0,
                "nesting_weight": 0.5,
                "first_pass_weight": 0.5,
                "energy_weight": -0.3
            }
        }
    """
    try:
        # Validate request
        if not request.plates:
            raise HTTPException(
                status_code=400,
                detail="At least one plate must be provided"
            )

        # Use default constraints and weights if not provided
        constraints = request.constraints or OptimizationConstraints()
        weights = request.weights or OptimizationWeights()

        # Execute optimization
        result = optimization_service.optimize_production(
            plates=request.plates,
            constraints=constraints,
            weights=weights
        )

        # Check for optimization errors
        if result.status == "failed":
            raise HTTPException(
                status_code=500,
                detail="Optimization failed due to internal error"
            )

        return result

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get(
    "/",
    tags=["Documentation"],
    summary="Root endpoint",
    description="Redirects to API documentation"
)
async def root() -> JSONResponse:
    """
    Root endpoint redirects to interactive API documentation.
    
    Returns:
        Redirect to /docs endpoint
    """
    return JSONResponse(
        status_code=200,
        content={
            "message": "Steel Plate Production Optimizer API",
            "docs": "/docs",
            "endpoints": {
                "health_check": "/ping",
                "optimization": "/optimize"
            }
        }
    )


if __name__ == "__main__":
    # Run development server
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
