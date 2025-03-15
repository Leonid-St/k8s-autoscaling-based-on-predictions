import pandas as pd
from io import StringIO
from flask import abort

def process_uploaded_file(request):
    file = request.files.get('file')
    if not file:
        abort(400, 'No file provided')
    
    file_type = file.filename.split('.')[-1].lower()
    
    try:
        if file_type == 'csv':
            df = pd.read_csv(StringIO(file.read().decode('utf-8')))
        elif file_type == 'json':
            df = pd.read_json(StringIO(file.read().decode('utf-8')))
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            abort(400, 'Unsupported file type')
    except Exception as e:
        abort(400, f'Error reading file: {str(e)}')

    if 'timestamp' not in df.columns:
        abort(400, 'File must contain timestamp column')

    return add_time_features(df)

def add_time_features(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'] >= 5
    df['minutes_since_midnight'] = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
    return df 
