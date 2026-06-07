import pandas as pd

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Generate useful features."""
    feature_data = data.copy()

    feature_data["Quarter"] = ((feature_data["Month"] - 1) // 3) + 1
    feature_data["Is_Q4"] = (feature_data["Quarter"] == 4).astype(int)

    start_year = feature_data["Year"].min()
    feature_data["Year_Month_Index"] = ((feature_data["Year"] - start_year) * 12 + feature_data["Month"])
    
    # Advanced Time-Series Features
    feature_data["Lag_1"] = feature_data["Estimated_Deliveries"].shift(1)
    feature_data["Lag_3"] = feature_data["Estimated_Deliveries"].shift(3)
    feature_data["Lag_6"] = feature_data["Estimated_Deliveries"].shift(6)
    
    feature_data["Rolling_Mean_3"] = feature_data["Estimated_Deliveries"].rolling(window=3).mean()
    feature_data["Rolling_Mean_6"] = feature_data["Estimated_Deliveries"].rolling(window=6).mean()
    
    feature_data["YoY_Growth"] = feature_data["Estimated_Deliveries"].pct_change(periods=12)
    
    if "Production" in feature_data.columns:
        feature_data["Production_Delivery_Ratio"] = feature_data["Production"] / (feature_data["Estimated_Deliveries"] + 1)
        
    feature_data = feature_data.bfill()

    return feature_data
