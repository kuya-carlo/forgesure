import sys
import os
from flask import Flask, request, jsonify
from pydantic import ValidationError

# Add parent directory to sys.path so imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from plate_service import PlateOptimizationService
    from models import PlateRequest, OptimizationConstraints, OptimizationWeights
except ImportError as e:
    # Log import error for debugging
    print(f"Import error: {e}", file=sys.stderr)
    raise


app = Flask(__name__)


@app.route('/', methods=['POST'])
def handler():
    """Vercel-compatible wrapper that calls the PlateOptimizationService.

    Expects JSON body matching the same schema used by the FastAPI endpoint.
    """
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"detail": "Invalid JSON body"}), 400

        plates_data = data.get('plates')
        if not plates_data:
            return jsonify({"detail": "At least one plate must be provided"}), 400

        try:
            plates = [PlateRequest(**p) for p in plates_data]
            constraints = OptimizationConstraints(**data.get('constraints', {}))
            weights = OptimizationWeights(**data.get('weights', {}))
        except ValidationError as e:
            return jsonify({"detail": str(e.errors())}), 400
        except Exception as e:
            return jsonify({"detail": str(e)}), 400

        service = PlateOptimizationService()
        result = service.optimize_production(plates, constraints, weights)

        # Pydantic v2 models expose model_dump(); fall back to dict-like access
        if hasattr(result, 'model_dump'):
            payload = result.model_dump()
        else:
            # best-effort fallback
            payload = result.__dict__

        return jsonify(payload), 200
    except Exception as e:
        return jsonify({"detail": f"Internal server error: {str(e)}"}), 500
