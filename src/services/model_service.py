# Clean Architecture: Service layer alias
# Re-export ModelService from existing implementation to avoid duplication.

from src.prediction.model_service import ModelService  # noqa: F401
