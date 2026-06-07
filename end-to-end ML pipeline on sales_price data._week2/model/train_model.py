import os
import sys
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Add parent directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import clean_data, cap_outliers
from utils.feature_engineering import create_features

class Config:
    TEST_SIZE = 0.20
    RANDOM_STATE = 42
    TARGET_COLUMN = "Estimated_Deliveries"
    CV_SPLITS = 5
    LEAKY_COLUMNS = ["Production", "Revenue", "Cumulative_Deliveries", "Delivery_Growth"]

def load_data():
    try:
        import kagglehub
        dataset_path = kagglehub.dataset_download("nalisha/tesla-ea-deliveries-and-production-data20152025")
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("No CSV file found in kagglehub path.")
        data = pd.read_csv(os.path.join(dataset_path, csv_files[0]))
        # Save to local data folder for future use
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'), exist_ok=True)
        data.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'tesla_data.csv'), index=False)
        return data
    except Exception as e:
        print(f"Failed to load from kagglehub: {e}. Looking for local data/tesla_data.csv")
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'tesla_data.csv')
        return pd.read_csv(data_path)

def evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions)
    }

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    
    print("Preprocessing data...")
    df_clean = clean_data(df)
    df_clean = cap_outliers(df_clean)
    df_features = create_features(df_clean)

    feature_columns = [
        col for col in df_features.columns
        if df_features[col].dtype != "object"
        and col != Config.TARGET_COLUMN
        and col not in Config.LEAKY_COLUMNS
    ]

    X = df_features[feature_columns]
    y = df_features[Config.TARGET_COLUMN]

    print("Splitting and scaling data...")
    split_index = int(len(df_features) * (1 - Config.TEST_SIZE))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=Config.RANDOM_STATE),
        "Random Forest": RandomForestRegressor(random_state=Config.RANDOM_STATE, n_estimators=100),
        "Gradient Boosting": GradientBoostingRegressor(random_state=Config.RANDOM_STATE),
        "XGBoost": xgb.XGBRegressor(random_state=Config.RANDOM_STATE, n_estimators=100)
    }

    print("Evaluating models...")
    results = []
    for model_name, model in models.items():
        res = evaluate_model(model, model_name, X_train, y_train, X_test, y_test)
        results.append(res)
        
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs'), exist_ok=True)
    results_df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'model_metrics.csv'), index=False)
    
    print("Tuning best model (Random Forest)...")
    parameter_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10],
        "min_samples_split": [2, 5]
    }
    tscv = TimeSeriesSplit(n_splits=Config.CV_SPLITS)
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=Config.RANDOM_STATE),
        param_grid=parameter_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("Saving artifacts...")
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    
    # Save Feature Importances
    fi_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": best_model.feature_importances_
    })
    fi_df.to_csv(os.path.join(outputs_dir, 'feature_importances.csv'), index=False)
    
    # Save Best Params
    import json
    with open(os.path.join(outputs_dir, 'best_params.json'), 'w') as f:
        json.dump(grid_search.best_params_, f, indent=4)
        
    joblib.dump(best_model, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tesla_model.pkl'))
    joblib.dump(scaler, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaler.pkl'))
    
    print("Training complete!")
