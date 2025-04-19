from metaflow import FlowSpec, step, Parameter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import mlflow
import os

class SpotifyTrainingFlow(FlowSpec):
    # Parameters for the flow
    test_size = Parameter('test_size', 
                         help='Size of test set',
                         default=0.2)
    
    random_state = Parameter('random_state',
                           help='Random seed for reproducibility',
                           default=42)
    
    @step
    def start(self):
        """Load and preprocess the data"""
        # Load the processed data
        self.df = pd.read_csv('data/save_data/spotify_processed.csv')
        
        # Define features and target
        self.features = ['danceability', 'energy', 'loudness', 'speechiness',
                        'acousticness', 'instrumentalness', 'liveness', 'valence',
                        'tempo', 'energy_danceability_ratio', 'loudness_energy_ratio']
        self.target = 'popularity'
        
        # Handle infinite values in derived features
        self.df['energy_danceability_ratio'] = self.df['energy_danceability_ratio'].replace([np.inf, -np.inf], np.nan)
        self.df['loudness_energy_ratio'] = self.df['loudness_energy_ratio'].replace([np.inf, -np.inf], np.nan)
        
        # Handle NaN values
        imputer = SimpleImputer(strategy='mean')
        X = self.df[self.features]
        y = self.df[self.target]
        
        # Impute missing values
        X_imputed = imputer.fit_transform(X)
        self.X = pd.DataFrame(X_imputed, columns=X.columns)
        self.y = y
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        print("Data loaded, cleaned, and split successfully")
        self.next(self.train_rf, self.train_lr)  # Split into two parallel branches
    
    @step
    def train_rf(self):
        """Train Random Forest model"""
        self.model_name = 'random_forest'
        self.model = RandomForestRegressor(random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train)
        
        # Calculate test score
        y_pred = self.model.predict(self.X_test)
        self.score = mean_squared_error(self.y_test, y_pred)
        
        self.next(self.choose_model)
    
    @step
    def train_lr(self):
        """Train Linear Regression model"""
        self.model_name = 'linear_regression'
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        
        # Calculate test score
        y_pred = self.model.predict(self.X_test)
        self.score = mean_squared_error(self.y_test, y_pred)
        
        self.next(self.choose_model)
    
    @step
    def choose_model(self, inputs):
        """Choose the best model based on test performance"""
        # Initialize MLFlow with SQLite backend
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        
        # Create experiment if it doesn't exist
        experiment_name = "spotify-popularity-prediction_lab6"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        
        # Set the experiment
        mlflow.set_experiment(experiment_name)
        
        # Find the best model from all inputs
        best_score = float('inf')
        best_model = None
        best_model_name = None
        
        for inp in inputs:
            if inp.score < best_score:
                best_score = inp.score
                best_model = inp.model
                best_model_name = inp.model_name
        
        # Log the best model to MLFlow
        with mlflow.start_run():
            mlflow.log_metric("mse", best_score)
            mlflow.log_param("model_type", best_model_name)
            mlflow.sklearn.log_model(best_model, "model")
            
            # Register the model
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/model",
                "spotify-popularity-model"
            )
        
        self.model = best_model
        self.model_name = best_model_name
        self.best_score = best_score
        self.next(self.end)
    
    @step
    def end(self):
        """End the flow"""
        print(f"Best model: {self.model_name}")
        print(f"Best MSE: {self.best_score}")

if __name__ == '__main__':
    SpotifyTrainingFlow() 