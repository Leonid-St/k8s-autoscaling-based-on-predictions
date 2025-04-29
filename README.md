# k8s-autoscaling-based-on-predictions

### for mvp  simple polynomial, SARIMA, XGBOOST  go to - folder autoscaler_predictor_model:
#### Autoscaling Endpoints:
- `/metrics/<model_type>` - комбинированная метрика (макс. из CPU/RAM)
- `/metrics/<model_type>/cpu` - метрика только по CPU
- `/metrics/<model_type>/memory` - метрика только по RAM



k8s-autoscale-controller