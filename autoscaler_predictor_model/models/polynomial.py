from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class PolynomialModel:
    def __init__(self):
        self.cpu_model = None
        self.memory_model = None
        self.requests_model = None

    def fit(self, df, features):
        if 'cpu' in df.columns:
            self.cpu_model = make_pipeline(
                PolynomialFeatures(4), 
                LinearRegression()
            ).fit(features, df['cpu'])
        
        if 'memory' in df.columns:
            self.memory_model = make_pipeline(
                PolynomialFeatures(4),
                LinearRegression()
            ).fit(features, df['memory'])
        
        if 'requests' in df.columns:
            self.requests_model = make_pipeline(
                PolynomialFeatures(4),
                LinearRegression()
            ).fit(features, df['requests'])

    def predict(self, features, predict_type):
        if predict_type == 'resource':
            return {
                'cpu': self.cpu_model.predict(features)[0],
                'memory': self.memory_model.predict(features)[0]
            }
        elif predict_type == 'requests':
            return {
                'requests': self.requests_model.predict(features)[0]
            }
        return {} 
