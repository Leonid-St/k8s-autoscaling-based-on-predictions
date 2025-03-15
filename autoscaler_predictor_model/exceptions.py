class ModelTrainingError(Exception):
    """Ошибка во время обучения модели"""
    
class PredictionError(Exception):
    """Ошибка во время предсказания"""
    
class DataValidationError(ValueError):
    """Ошибка валидации входных данных"""
