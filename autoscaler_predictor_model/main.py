import asyncio
import json
import os

from fastapi import FastAPI, Depends, HTTPException, status
import requests
from datetime import datetime, timedelta
import warnings

from pydantic import BaseModel
from urllib3.exceptions import NotOpenSSLWarning

from models.polynomial import PolynomialModel
from metrics.metrics_collector import MetricsCollector
from config import EnvConfig
from metrics.metrics_fetcher import MetricsFetcher
from metrics.metrics_fetcher_victoria import VictoriaMetricsFetcher
from metrics.metrics_fetcher_local_test import MetricsFetcherLocalTest
from models.sarima import SARIMAModel
from models.xgboost import XGBoostModel
from services.predictor_service import PredictorService
from services.teacher_service import TeacherService
from storage.storage_factory import StorageFactory
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from storage.storage_service import StorageService
import uvicorn
from contextlib import asynccontextmanager

from services.comparator_service import ComparatorService
from fastapi.responses import JSONResponse
from services.scaler_service import ScalerService
import logging

logger = logging.getLogger("uvicorn.error")



from pytz import utc

scheduler = AsyncIOScheduler(timezone=utc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация конфигурации
    env_config = EnvConfig()

    # 0 - Инициализация хранилища
    storage = StorageFactory.create_storage(env_config)

    # 1 - fetch metrics
    use_local_fetcher = os.getenv("USE_LOCAL_FETCHER", "false").lower() == "true"

    fetcher = VictoriaMetricsFetcher(env_config.url_for_metrics)


    collector = MetricsCollector(
        uuid=env_config.uuid_node,
        fetcher=fetcher,
        storage=storage,
    )

    # Инициализация моделей
    models = {
        'polynomial': PolynomialModel(),
        'xgboost': XGBoostModel(),
        'sarima': SARIMAModel(data_retention='4H')
    }

    teacher_service = TeacherService(
        model=models['xgboost'],
        storage=storage,
        node_id=env_config.uuid_node
    )


    # Регистрируем TeacherService как наблюдателя



    collector.add_observer(teacher_service.on_new_data)

    # Инициализация PredictorService
    predictor_service = PredictorService(
        model=models['xgboost'],
        storage=storage,
        node_id=env_config.uuid_node
    )

    # Регистрируем PredictorService как наблюдателя в TeacherService
    teacher_service.add_observer(predictor_service.on_model_updated)

    # Инициализация ComparatorService
    comparator_service = ComparatorService(
        storage=storage,
        node_id=env_config.uuid_node
    )

    # Регистрируем ComparatorService как наблюдателя в PredictorService
    predictor_service.add_observer(comparator_service.compare_and_save_errors)

    scheduler.add_job(collector.collect, 'interval', seconds=60)

    # Инициализация ScalerService
    scaler_service = ScalerService(
        storage=storage,
        node_id=env_config.uuid_node,
        keda_scaler_address="http://autoscaler-service:5001"
    )
    # scheduler.add_job(scaler_service.scale, 'interval', minutes=1)

    scheduler.start()

    yield

    # Закрытие соединения с базой данных
    if hasattr(storage, 'conn') and not storage.conn.closed:
        storage.conn.close()
    scheduler.shutdown()


app = FastAPI(lifespan=lifespan)


async def get_storage_service() -> StorageService:
    env_config = EnvConfig()
    return StorageFactory.create_storage(env_config)




#if __name__ == "__main__":






@app.get("/keda-metrics")
async def keda_metrics(storage_service: StorageService = Depends(get_storage_service)):
    """
    Endpoint для Keda, возвращает последнее предсказанное значение CPU из базы данных.
    """
    latest_prediction = await storage_service.get_latest_prediction(
        node=EnvConfig().uuid_node,
    )

    if latest_prediction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Предсказания не найдены")

    cpu_prediction = latest_prediction.metrics.get('cpu')

    if cpu_prediction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CPU предсказание не найдено")

    logger.info("metric has been pulling by keda:",latest_prediction)
    return JSONResponse({"cpu": cpu_prediction})

async def get_predictor_service() -> PredictorService:
    env_config = EnvConfig()
    storage = StorageFactory.create_storage(env_config)
    models = {
        'xgboost': XGBoostModel(),
        'sarima': SARIMAModel(data_retention='4H'),
        'polynomial': PolynomialModel()
    }
    return PredictorService(
        model=models['xgboost'],
        storage=storage,
        node_id=env_config.uuid_node
    )

