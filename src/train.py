import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib
import json
import os
from utils import load_data, get_features_and_target

NAME = "M.Ruthwik"
ROLLNO = "2022BCS0135"
mlflow.set_tracking_uri("file:./mlruns")
def train(data_path: str, model_type: str, subset_features: bool, run_desc: str):
    mlflow.set_experiment(f"{ROLLNO}_experiment")
    
    with mlflow.start_run(run_name=run_desc):
        df = load_data(data_path)
        
        feature_cols = None
        if subset_features:
            # Drop some features as a form of feature selection (using subset)
            feature_cols = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'alcohol', 'sulphates']
        
        target_col = 'quality'
            
        X, y = get_features_and_target(df, target_col, feature_cols)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == 'lr':
            model = LinearRegression()
        elif model_type == 'lr_hyper':
            # linear regression with no intercept as alternate hyperparameter
            model = LinearRegression(fit_intercept=False)
        elif model_type == 'rf':
            model = RandomForestRegressor(n_estimators=50, random_state=42)
        else:
            raise ValueError("Unsupported model_type")
            
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("subset_features", subset_features)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        mlflow.sklearn.log_model(model, "model")
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/model.joblib')
        
        metrics = {
            "mse": mse,
            "r2": r2,
            "name": NAME,
            "roll_no": ROLLNO
        }
        
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
        print(f"Run '{run_desc}' completed. MSE: {mse:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--model_type", type=str, required=True, choices=['lr', 'lr_hyper', 'rf'])
    parser.add_argument("--subset_features", action='store_true', help="Use feature selection")
    parser.add_argument("--run_desc", type=str, default="run")
    args = parser.parse_args()
    
    train(args.data_path, args.model_type, args.subset_features, args.run_desc)
