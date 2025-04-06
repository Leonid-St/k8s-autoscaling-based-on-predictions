import pandas as pd
from io import StringIO
from flask import abort
from exceptions import DataValidationError


def process_uploaded_file(file):
    try:
        if not file or file.filename == '':
            raise DataValidationError("No file uploaded")

        # if file.content_length > 1024 * 1024:  # Ограничение 1MB
        #     raise DataValidationError("File size exceeds 1MB limit")

        file_type = file.filename.split('.')[-1].lower()

        content = file.file.read().decode('utf-8')
        
        if file_type == 'csv':
            df = pd.read_csv(StringIO(content))
        elif file_type == 'json':
            df = pd.read_json(StringIO(content))
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
        else:
            raise DataValidationError('Unsupported file type')

        if 'timestamp' not in df.columns:
            raise DataValidationError('File must contain timestamp column')

        return add_time_features(df)

    except Exception as e:
        raise DataValidationError(str(e))


def add_time_features(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'] >= 5
    df['minutes_since_midnight'] = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
    return df
