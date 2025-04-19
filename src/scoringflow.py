from metaflow import FlowSpec, step, Parameter, JSONType
import pandas as pd
import numpy as np
import mlflow
import json

class SpotifyScoringFlow(FlowSpec):
    # Parameter for input data
    input_data = Parameter('input_data',
                         help='Input data in JSON format',
                         type=JSONType,
                         required=True)
    
    @step
    def start(self):
        """Load the input data and prepare it for prediction"""
        # Convert input JSON to DataFrame
        self.input_df = pd.DataFrame([self.input_data])
        
        # Define required features
        self.features = ['danceability', 'energy', 'loudness', 'speechiness',
                        'acousticness', 'instrumentalness', 'liveness', 'valence',
                        'tempo']
        
        # Calculate derived features
        self.input_df['energy_danceability_ratio'] = self.input_df['energy'] / self.input_df['danceability']
        self.input_df['loudness_energy_ratio'] = self.input_df['loudness'] / self.input_df['energy']
        
        # Ensure all required features are present
        missing_features = set(self.features) - set(self.input_df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        self.next(self.predict)
    
    @step
    def predict(self):
        """Load the model and make predictions"""
        # Initialize MLFlow
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        
        # Load the latest version of the registered model
        model_uri = f"models:/spotify-popularity-model/latest"
        self.model = mlflow.sklearn.load_model(model_uri)
        
        # Prepare features for prediction
        X = self.input_df[self.features + ['energy_danceability_ratio', 'loudness_energy_ratio']]
        
        # Make prediction
        self.prediction = self.model.predict(X)[0]
        
        self.next(self.end)
    
    @step
    def end(self):
        """Output the prediction"""
        print(f"Predicted popularity score: {self.prediction:.2f}")

if __name__ == '__main__':
    SpotifyScoringFlow() 