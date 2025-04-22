from metaflow import FlowSpec, step, Parameter, kubernetes, retry, catch, conda_base


@conda_base(python="3.10", libraries={
    "pandas": "1.5.3",
    "scikit-learn": "1.2.2",
    "mlflow": "2.2.2",
    "numpy": "1.23.5",
    "gcsfs": "2023.6.0",
    "databricks-cli": "0.17.6"
})
class DiabetesGCPTrainFlow(FlowSpec):
    """
    A Metaflow pipeline for training and registering a RandomForestRegressor
    model on the diabetes dataset using MLflow and GCP.
    """

    mlflow_uri = Parameter(
        'mlflow_uri',
        default="https://dockservice-98117875378.us-west2.run.app/",
        help="MLflow tracking server URI"
    )
    n_trials = Parameter(
        'trials',
        default=5,
        help="Number of hyperparameter trials to run"
    )

    @catch(var="load_error")
    @step
    def start(self):
        """
        Load the diabetes dataset and prepare train/test splits.
        """
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import train_test_split

        # Load diabetes dataset
        diabetes = load_diabetes(return_X_y=True)
        X, y = diabetes

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Data loaded: {len(X)} records")
        self.next(self.train_models)

    @kubernetes(cpu=2, memory=4096)
    @catch(var="train_error")
    @step
    def train_models(self):
        """
        Train multiple models with different hyperparameters.
        """
        import mlflow
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import ParameterSampler
        from sklearn.metrics import mean_squared_error
        import numpy as np

        # Set up MLflow tracking
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment("diabetes-regression-gcp")

        # Define parameter grid
        param_dist = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
        }

        # Sample parameters
        param_list = list(ParameterSampler(
            param_distributions=param_dist,
            n_iter=self.n_trials,
            random_state=42
        ))

        # Train models
        best_score, best_model, best_params, best_run_id = float('inf'), None, None, None

        for i, params in enumerate(param_list):
            with mlflow.start_run(run_name=f"run_{i}") as run:
                # Train model
                model = RandomForestRegressor(random_state=42, **params)
                model.fit(self.X_train, self.y_train)

                # Evaluate
                y_pred = model.predict(self.X_test)
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)

                # Log to MLflow
                mlflow.log_params(params)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("rmse", rmse)

                print(f"Run {i}: RMSE = {rmse:.4f}")

                # Track best model
                if rmse < best_score:
                    best_score = rmse
                    best_model = model
                    best_params = params
                    best_run_id = run.info.run_id

        # Store the best model
        self.best_model = best_model
        self.best_score = best_score
        self.best_run_id = best_run_id

        print(f"Best RMSE: {best_score:.4f}")
        self.next(self.register_model)

    
    @retry(times=2)
    @catch(var="register_error")
    @step
    def register_model(self):
        """
        Register the best model in MLflow Model Registry.
        """
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(self.mlflow_uri)

        with mlflow.start_run(run_name="register_best_model"):
            mlflow.log_param("source_run_id", self.best_run_id)
            mlflow.sklearn.log_model(
                sk_model=self.best_model,
                artifact_path="model",
                registered_model_name="diabetes-rf-regressor"
            )

        print("Best model registered to MLflow")
        self.next(self.end)

    @step
    def end(self):
        """
        Final step to summarize the flow results.
        """
        print("Training and registration completed.")
        print(f"Best RMSE: {self.best_score:.4f}")
        print(f"Best model registered to MLflow")


if __name__ == '__main__':
    DiabetesGCPTrainFlow()
