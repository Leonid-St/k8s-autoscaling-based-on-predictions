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
export PYENV_ROOT="$HOME/.pyenv"

[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"

eval "$(pyenv init - bash)"
```

env package :
```
brew install libomp
```
```bash
deactivate
rm -rf autoscaler_predictor_model/venv
```

```bash
python3 -m venv venv
source venv/bin/activate
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

```bash
uvicorn main:app --reload
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


### Metric middle accuracy over time
```
curl "http://localhost:5001/metrics/accuracy?start_date=2024-01-01&end_date=2024-03-01"
```

### historical error

```
curl "http://localhost:5001/metrics/errors?start_date=2024-01-01&end_date=2024-03-01"
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


### Install InfluxDB on ubuntu 24.04

https://docs.influxdata.com/influxdb/v2/install/?t=Linux#choose-the-influxdata-key-pair-for-your-os-version

```bash
curl --location -O \
https://download.influxdata.com/influxdb/releases/influxdb2-2.7.11_linux_amd64.tar.gz

tar xvzf ./influxdb2-2.7.11_linux_amd64.tar.gz

sudo cp ./influxdb2-2.7.11/usr/bin/influxd /usr/local/bin/


./influxdb2-2.7.11/usr/bin/influxd

#Recommended 
chmod 0750 ~/.influxdbv2


```


#### Start InfluxDB
```bash
influxd http-bind-address=:8086 reporting-disabled=false
```


### Install the latest PostgreSQL packages with TimeScaleDB

```bash
sudo apt install gnupg postgresql-common apt-transport-https lsb-release wget
```
Run the PostgreSQL package setup script
```bash
sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh
```
If you want to do some development on PostgreSQL, add the libraries:
```bash
sudo apt install postgresql-server-dev-17
```
Add the TimescaleDB package


Ubuntu
```bash
echo "deb https://packagecloud.io/timescale/timescaledb/debian/ $(lsb_release -c -s) main" | sudo tee /etc/apt/sources.list.d/timescaledb.list
```
Install the TimescaleDB GPG key
```bash
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/timescaledb.gpg
```


Update your local repository list
```bash
sudo apt update
```
Install TimescaleDB
```bash
sudo apt install timescaledb-2-postgresql-17 postgresql-client-17
```
To install a specific TimescaleDB release, set the version. For example:
```bash
sudo apt-get install timescaledb-2-postgresql-14='2.6.0*' timescaledb-2-loader-postgresql-14='2.6.0*'
```
Older versions of TimescaleDB may not support all the OS versions listed on this page.

Tune your PostgreSQL instance for TimescaleDB
```bash
sudo timescaledb-tune
```
This script is included with the timescaledb-tools package when you install TimescaleDB. For more information, see configuration.

Restart PostgreSQL
```bash
sudo systemctl restart postgresql
```
Login to PostgreSQL as postgres
```bash
sudo -u postgres psql
```
You are in the psql shell.

Set the password for postgres
```bash
\password postgres
```
When you have set the password, type \q to exit psql.

#### Add the TimescaleDB extension to your database


Connect to a database on your PostgreSQL instance

In PostgreSQL, the default user and database are both postgres. To use a different database, set <database-name> to the name of that database:
```bash
psql -d "postgres://<username>:<password>@<host>:<port>/<database-name>"
```
Add TimescaleDB to the database
```bash
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
```
Check that TimescaleDB is installed
```bash
\dx
```
You see the list of installed extensions:
```bash
List of installed extensions
Name     | Version |   Schema   |                                      Description                                      
-------------+---------+------------+---------------------------------------------------------------------------------------
plpgsql     | 1.0     | pg_catalog | PL/pgSQL procedural language
timescaledb | 2.17.2  | public     | Enables scalable inserts and complex queries for time-series data (Community Edition)
```
Press q to exit the list of extensions.
