# autoscaler_predictor_model
The data synthesizer, forecaster and predictor model server used together with KEDA scaler or cluster autoscaler to achieve cluster autoscaling with diurnal pattern workload
```
Ensure that the cluster has the following installed:
kube-state-metrics
node-exporter
prometheus-operator
```
## How to use the data synthesizer
### Set up python virtual environment

```bash
export LDFLAGS="-L$(brew --prefix openssl)/lib -L$(brew --prefix readline)/lib -L$(brew --prefix zlib)/lib"
export CPPFLAGS="-I$(brew --prefix openssl)/include -I$(brew --prefix readline)/include -I$(brew --prefix zlib)/include"
export PKG_CONFIG_PATH="$(brew --prefix openssl)/lib/pkgconfig:$(brew --prefix readline)/lib/pkgconfig:$(brew --prefix zlib)/lib/pkgconfig"
```

```bash
brew update
brew upgrade
brew install openssl readline sqlite3 xz zlib tcl-tk
pyenv install 3.9.7
pyenv local 3.9.7
 ```

```bash
# Для bash
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init --path)"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile

# Для zsh
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```


```bash
deactivate
rm -rf autoscaler_predictor_model/venv
```

```bash
python3 -m venv autoscaler_predictor_model/venv
source autoscaler_predictor_model/venv/bin/activate
pip install -r requirements.txt
```

### Usage Instructions:
1. Configure the data synthesizer:
- Edit the `config.ini` file to set the peak and valley CPU/memory usage or requests/s load. You can also configure
the number of weeks to generate the data, the peak hour in a day and the variations of usage/load across the same time in a day.
2. Generate Data:
```bash
python data_synthesizer.py
```
3. The data is generated in the `data` folder. The data is generated in CSV format with the following columns:
- timestamp: The timestamp in milliseconds since epoch
- cpu: The CPU usage in millicores
- memory: The memory usage in bytes
- requests: The number of requests per second
The csv file name is in the format: `<type>-<year>-<month>-<day>-<hour>-<minute>.csv` and can be fed into the model server to fit the model.
The json file name is in the format: `<type>-<year>-<month>-<day>-<hour>-<minute>.json` and can be fed into the model server to fit the model or to forecast the next datapoint.

## How to use the model server
### Build and Run the Docker image
```bash
podman machine init
export DOCKER_HOST='unix:///var/folders/2v/7rm59ty53ld26_byy1x3z1g80000gn/T/podman/podman-machine-default-api.sock'

podman machine start
podman build -t autoscaler-prediction-model-server .
podman run -p 5001:5001 autoscaler-prediction-model-server
```

### Usage Instructions:
1. Fit Models:
- POST `/fit/<model_type>` (polynomial/xgboost/sarima)
- Пример: `curl -X POST http://localhost:5001/fit/xgboost -F 'file=@data.csv'`

2. Get Predictions:
- GET `/predict/<model_type>?timestamp=<ISO timestamp>`
- Пример: `curl http://localhost:5001/predict/sarima?timestamp=2024-03-15T14:30:00`

3. Get Forecasts:

- Send a POST request to /forecast with a csv file or json object that contains any length of time series data.
- The server returns the predicted cpu/memory or requests value(s) based on SARIMA algorithm running on the given data.
Example `curl` command to get forecasts from a csv file:
```bash
 curl -X POST http://127.0.0.1:5001/forecast \
-H "Content-Type: multipart/form-data" \
-F "file=@./data/requests-25-03-05-11-51.csv"
```

Example `curl` command to get forecasts from a json object:
```bash
curl -X POST http://127.0.0.1:5001/forecast \
-H "Content-Type: application/json" \
-d '[{"timestamp":1700092800000,"requests":279},{"timestamp":1700093700000,"requests":257},{"timestamp":1700094600000,"requests":230}]'
```


#### Notes:
- The fit-model endpoint expects a CSV file with two columns: timestamp and requests or three columes: timestamp, cpu and memory. Sample data can be found in the `data` folder.
- The predict endpoint requires a timestamp in a recognizable datetime format.
- The model fitting is simplistic and assumes a polynomial model. You may need to adjust the model fitting part based on your specific requirements.

### Autoscaling Endpoints:
- `/metrics/<model_type>` - combined metric (ma from CPU/RAM)
- `/metrics/<model_type>/cpu` - CPU-only metrics
- `/metrics/<model_type>/memory` - RAM-only metrics

example of response :

```json
{
  "value": 72.34,
  "timestamp": "2024-03-15T14:30:00"
}
```


### Metric accuracy 
```
curl "http://localhost:5001/metrics/accuracy?start_date=2024-01-01&end_date=2024-03-01"
```


## Automatic Prediction and Model Updates

The application now automatically:
1. Collects metrics from Prometheus every minute
2. Updates the XGBoost model with new data
3. Makes predictions 5 minutes ahead
4. Stores predictions in a file for quick access

###  API Endpoints for Latest 5 min prediction

#### Get Latest Prediction

```
GET /predict/latest
```

```
{
"timestamp": "2024-03-15T14:35:00",
"cpu": 0.75,
"memory": 0.65
}
```


### Storage Details
- Predictions are stored in `./predictions/latest_prediction.parquet`
- Historical metrics are stored in `./metrics_data/` with daily files
- Data retention: 365 days for metrics, latest prediction only

### Automatic Workflow
The background scheduler performs these steps every minute:
1. Collects new metrics from Prometheus
2. Updates the XGBoost model with partial_fit
3. Makes a prediction 5 minutes ahead
4. Saves the prediction to file

### Error Handling
- Failed Prometheus queries are logged and retried
- Model update failures are logged but don't stop the process
- Missing prediction files return 404 status



example  ScaledObject KEDA :
```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: cpu-scaler
spec:
  scaleTargetRef:
    name: your-deployment
  triggers:
    - type: external-push
      metadata:
        scalerAddress: "autoscaler-service:5001"
        url: "http://autoscaler-service:5001/metrics/xgboost/cpu"
        threshold: "70"
---
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: memory-scaler
spec:
  scaleTargetRef:
    name: your-deployment
  triggers:
    - type: external-push
      metadata:
        scalerAddress: "autoscaler-service:5001"
        url: "http://autoscaler-service:5001/metrics/xgboost/memory"
        threshold: "75"
---
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: combined-scaler
spec:
  scaleTargetRef:
    name: your-deployment
  triggers:
    - type: external-push
      metadata:
        scalerAddress: "autoscaler-service:5001"
        url: "http://autoscaler-service:5001/metrics/xgboost"
        threshold: "80"

```

