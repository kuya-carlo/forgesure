"""
Comprehensive unit and integration tests for the API.

Tests cover:
- PlateDataProcessor KPI calculations
- PlateOptimizer LP formulation and solving
- API endpoints (ping, optimize)
- Error handling and edge cases
"""

import pytest
from fastapi.testclient import TestClient
from main import app
from models import (
    PlateRequest,
    OptimizationConstraints,
    OptimizationWeights,
    OptimizationRequest
)
from plate_service import (
    PlateDataProcessor,
    PlateOptimizer,
    PlateOptimizationService
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def test_client():
    """Provide FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_plates():
    """Provide sample plate data for testing."""
    return [
        PlateRequest(
            id='P001',
            thickness=12,
            length=8000,
            width=2000,
            holes=4,
            beam_size='20x12',
            price=5000
        ),
        PlateRequest(
            id='P002',
            thickness=16,
            length=10000,
            width=2500,
            holes=6,
            beam_size='27x16',
            price=7000
        ),
        PlateRequest(
            id='P003',
            thickness=20,
            length=12000,
            width=3000,
            holes=8,
            beam_size='30x16',
            price=9000
        ),
        PlateRequest(
            id='P004',
            thickness=10,
            length=6000,
            width=1500,
            holes=2,
            beam_size='15x12',
            price=3500
        ),
    ]


@pytest.fixture
def default_constraints():
    """Provide default optimization constraints."""
    return OptimizationConstraints()


@pytest.fixture
def default_weights():
    """Provide default objective function weights."""
    return OptimizationWeights()


# ============================================================================
# TESTS: PlateDataProcessor
# ============================================================================

class TestPlateDataProcessor:
    """Tests for PlateDataProcessor class."""

    def test_processor_initialization(self, sample_plates, default_constraints):
        """Test processor can be initialized with plates and constraints."""
        processor = PlateDataProcessor(sample_plates, default_constraints)
        assert processor.df is None
        assert len(processor.plates) == 4

    def test_process_plates_creates_dataframe(self, sample_plates, default_constraints):
        """Test that process_plates creates a valid DataFrame."""
        processor = PlateDataProcessor(sample_plates, default_constraints)
        df = processor.process_plates()

        assert df is not None
        assert len(df) == 4
        assert 'id' in df.columns
        assert all(col in df.columns for col in processor.PROCESS_TIME_COLUMNS)

    def test_geometric_kpi_calculations(self, sample_plates, default_constraints):
        """Test geometric KPI calculations are valid."""
        processor = PlateDataProcessor(sample_plates, default_constraints)
        df = processor.process_plates()

        # Check nesting utilization is between 0 and 1
        assert all(0 <= util <= 1 for util in df['nesting_utilization'])

        # Check weight calculations
        assert all(weight > 0 for weight in df['beam_weight_ton'])

        # Check volume is calculated correctly
        for _, row in df.iterrows():
            expected_volume = row['length']/1000 * row['width']/1000 * row['thickness']/1000
            assert abs(row['beam_volume_m3'] - expected_volume) < 0.001

    def test_energy_calculations(self, sample_plates, default_constraints):
        """Test energy consumption calculations."""
        processor = PlateDataProcessor(sample_plates, default_constraints)
        df = processor.process_plates()

        # Energy should be positive
        assert all(energy > 0 for energy in df['energy_total_kWh'])

        # Energy intensity should be reasonable
        assert all(intensity > 0 for intensity in df['energy_intensity_kWh_per_ton'])

    def test_cost_calculations(self, sample_plates, default_constraints):
        """Test cost calculations."""
        processor = PlateDataProcessor(sample_plates, default_constraints)
        df = processor.process_plates()

        # All costs should be positive
        assert all(cost > 0 for cost in df['material_cost'])
        assert all(cost > 0 for cost in df['energy_cost'])
        assert all(cost > 0 for cost in df['labor_cost'])

    def test_profit_calculations(self, sample_plates, default_constraints):
        """Test profit calculations."""
        processor = PlateDataProcessor(sample_plates, default_constraints)
        df = processor.process_plates()

        # Profit should be price - costs
        for _, row in df.iterrows():
            expected_profit = (
                row['price'] -
                (row['material_cost'] + row['energy_cost'] + row['labor_cost'])
            )
            assert abs(row['profit'] - expected_profit) < 0.01

    def test_get_total_process_time(self, sample_plates, default_constraints):
        """Test total process time retrieval."""
        processor = PlateDataProcessor(sample_plates, default_constraints)
        df = processor.process_plates()

        # Get total time for first plate
        total_time = processor.get_total_process_time('P001')
        assert total_time > 0

        # Non-existent plate should return 0
        assert processor.get_total_process_time('NONEXISTENT') == 0.0


# ============================================================================
# TESTS: PlateOptimizer
# ============================================================================

class TestPlateOptimizer:
    """Tests for PlateOptimizer class."""

    def test_optimizer_initialization(self, sample_plates, default_constraints, default_weights):
        """Test optimizer initialization."""
        processor = PlateDataProcessor(sample_plates, default_constraints)
        processor.process_plates()
        optimizer = PlateOptimizer(processor, default_weights)

        assert optimizer.model is None
        assert len(optimizer.plate_vars) == 0

    def test_optimization_runs_successfully(self, sample_plates, default_constraints, default_weights):
        """Test optimization completes without errors."""
        processor = PlateDataProcessor(sample_plates, default_constraints)
        processor.process_plates()
        optimizer = PlateOptimizer(processor, default_weights)

        results, obj_value = optimizer.optimize(default_constraints)

        # Should return valid results
        assert results is not None
        assert len(results) == 4
        assert obj_value is not None

    def test_optimization_respects_time_constraint(self, sample_plates, default_constraints, default_weights):
        """Test optimization respects shift time constraint."""
        processor = PlateDataProcessor(sample_plates, default_constraints)
        processor.process_plates()
        optimizer = PlateOptimizer(processor, default_weights)

        results, _ = optimizer.optimize(default_constraints)

        # Calculate total time for scheduled plates
        scheduled_plates = [pid for pid, val in results.items() if val == 1.0]
        total_time = sum(
            processor.get_total_process_time(pid) for pid in scheduled_plates
        )

        # Should not exceed shift hours
        assert total_time <= default_constraints.max_shift_hours + 0.01

    def test_optimization_respects_plate_limit(self, sample_plates, default_constraints, default_weights):
        """Test optimization respects max plates per shift constraint."""
        processor = PlateDataProcessor(sample_plates, default_constraints)
        processor.process_plates()
        optimizer = PlateOptimizer(processor, default_weights)

        results, _ = optimizer.optimize(default_constraints)

        # Count scheduled plates
        scheduled_count = sum(1 for val in results.values() if val == 1.0)

        # Should not exceed limit
        assert scheduled_count <= default_constraints.max_plates_per_shift


# ============================================================================
# TESTS: PlateOptimizationService
# ============================================================================

class TestPlateOptimizationService:
    """Tests for PlateOptimizationService class."""

    def test_service_optimization_workflow(self, sample_plates):
        """Test complete optimization workflow."""
        service = PlateOptimizationService()
        constraints = OptimizationConstraints()
        weights = OptimizationWeights()

        result = service.optimize_production(sample_plates, constraints, weights)

        assert result.status in ["success", "failed"]
        assert isinstance(result.scheduled_plates, list)
        assert isinstance(result.kpis, list)
        assert result.total_profit >= 0
        assert result.total_process_time >= 0

    def test_service_returns_valid_kpis(self, sample_plates):
        """Test service returns valid KPI data."""
        service = PlateOptimizationService()
        constraints = OptimizationConstraints()
        weights = OptimizationWeights()

        result = service.optimize_production(sample_plates, constraints, weights)

        if result.scheduled_plates:
            # Each scheduled plate should have KPIs
            assert len(result.kpis) == len(result.scheduled_plates)

            # Check KPI values are valid
            for kpi in result.kpis:
                assert kpi.id in result.scheduled_plates
                assert kpi.profit >= -1e6  # Allow negative profit
                assert 0 <= kpi.nesting_utilization <= 1
                assert 0.8 <= kpi.first_pass_yield <= 1.0


# ============================================================================
# TESTS: API ENDPOINTS
# ============================================================================

class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""

    def test_ping_endpoint(self, test_client):
        """Test ping endpoint returns 200 and correct structure."""
        response = test_client.get("/ping")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_root_endpoint(self, test_client):
        """Test root endpoint returns documentation info."""
        response = test_client.get("/")
        assert response.status_code == 200
        assert "docs" in response.json()

    def test_optimize_endpoint_success(self, test_client, sample_plates):
        """Test optimize endpoint with valid request."""
        request_data = {
            "plates": [
                {
                    "id": p.id,
                    "thickness": p.thickness,
                    "length": p.length,
                    "width": p.width,
                    "holes": p.holes,
                    "beam_size": p.beam_size,
                    "price": p.price
                }
                for p in sample_plates
            ]
        }

        response = test_client.post("/optimize", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] in ["success", "failed"]
        assert "scheduled_plates" in data
        assert "kpis" in data
        assert "total_profit" in data

    def test_optimize_endpoint_empty_plates(self, test_client):
        """Test optimize endpoint rejects empty plates list."""
        request_data = {"plates": []}

        response = test_client.post("/optimize", json=request_data)
        assert response.status_code == 400

    def test_optimize_endpoint_with_constraints(self, test_client, sample_plates):
        """Test optimize endpoint with custom constraints."""
        request_data = {
            "plates": [
                {
                    "id": p.id,
                    "thickness": p.thickness,
                    "length": p.length,
                    "width": p.width,
                    "holes": p.holes,
                    "beam_size": p.beam_size,
                    "price": p.price
                }
                for p in sample_plates
            ],
            "constraints": {
                "max_shift_hours": 6.0,
                "max_plates_per_shift": 2,
                "material_cost_per_ton": 500,
                "energy_cost_per_kwh": 5,
                "labor_cost_per_hour": 50
            }
        }

        response = test_client.post("/optimize", json=request_data)
        assert response.status_code == 200
        assert response.json()["total_process_time"] <= 6.0

    def test_optimize_endpoint_with_weights(self, test_client, sample_plates):
        """Test optimize endpoint with custom weights."""
        request_data = {
            "plates": [
                {
                    "id": p.id,
                    "thickness": p.thickness,
                    "length": p.length,
                    "width": p.width,
                    "holes": p.holes,
                    "beam_size": p.beam_size,
                    "price": p.price
                }
                for p in sample_plates
            ],
            "weights": {
                "profit_weight": 2.0,
                "nesting_weight": 0.2,
                "first_pass_weight": 0.2,
                "energy_weight": -0.1
            }
        }

        response = test_client.post("/optimize", json=request_data)
        assert response.status_code == 200


# ============================================================================
# TESTS: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_plate_optimization(self, test_client):
        """Test optimization with only one plate."""
        request_data = {
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
            ]
        }

        response = test_client.post("/optimize", json=request_data)
        assert response.status_code == 200

    def test_very_tight_constraints(self, test_client, sample_plates):
        """Test optimization with very restrictive constraints."""
        request_data = {
            "plates": [
                {
                    "id": p.id,
                    "thickness": p.thickness,
                    "length": p.length,
                    "width": p.width,
                    "holes": p.holes,
                    "beam_size": p.beam_size,
                    "price": p.price
                }
                for p in sample_plates
            ],
            "constraints": {
                "max_shift_hours": 0.5,  # Very tight
                "max_plates_per_shift": 1
            }
        }

        response = test_client.post("/optimize", json=request_data)
        assert response.status_code == 200

    def test_negative_profit_plates(self, test_client):
        """Test optimization includes plates even with negative profit if beneficial."""
        request_data = {
            "plates": [
                {
                    "id": "P_EXPENSIVE",
                    "thickness": 50,
                    "length": 20000,
                    "width": 5000,
                    "holes": 20,
                    "beam_size": "50x50",
                    "price": 100  # Very low price
                }
            ]
        }

        response = test_client.post("/optimize", json=request_data)
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
