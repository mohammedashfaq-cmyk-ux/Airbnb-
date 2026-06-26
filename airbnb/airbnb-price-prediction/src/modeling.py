import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def prepare_features(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the data into train and test sets, and applies column transformations:
    - One-hot encoding for categorical variables (neighbourhood_group, room_type)
    - Scaling for numerical variables
    
    Returns:
    - preprocessor: Fitted sklearn ColumnTransformer
    - X_train_proc, X_test_proc: Processed features (numpy arrays)
    - y_train, y_test: Target variables (pandas series of log_price)
    - feature_names: List of feature names after processing
    """
    # Define features and target
    num_features = [
        'latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 
        'reviews_per_month', 'calculated_host_listings_count', 
        'availability_365', 'name_length', 'days_since_last_review'
    ]
    cat_features = ['neighbourhood_group', 'room_type']
    
    X = df[num_features + cat_features]
    y = df['log_price']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Build the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ])
    
    # Fit and transform
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_cols = cat_encoder.get_feature_names_out(cat_features).tolist()
    feature_names = num_features + encoded_cat_cols
    
    return preprocessor, X_train_proc, X_test_proc, y_train, y_test, feature_names

def train_all_models(X_train, y_train):
    """
    Trains multiple regression models: Linear Regression, Random Forest, and Gradient Boosting.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"Successfully trained {name}")
        
    return trained_models

def evaluate_predictions(model, X_test, y_test):
    """
    Evaluates predictions in both log scale and original dollar scale:
    - Log Scale metrics (RMSE, MAE, R2)
    - Dollar Scale metrics (RMSE, MAE, R2) by taking expm1 of values
    """
    # Predict (which returns log prices)
    y_pred_log = model.predict(X_test)
    
    # Dollar transformation
    y_pred_dollar = np.expm1(y_pred_log)
    y_true_dollar = np.expm1(y_test)
    
    # Log metrics
    mse_log = mean_squared_error(y_test, y_pred_log)
    rmse_log = np.sqrt(mse_log)
    mae_log = mean_absolute_error(y_test, y_pred_log)
    r2_log = r2_score(y_test, y_pred_log)
    
    # Dollar metrics (more business-interpretable)
    mse_dollar = mean_squared_error(y_true_dollar, y_pred_dollar)
    rmse_dollar = np.sqrt(mse_dollar)
    mae_dollar = mean_absolute_error(y_true_dollar, y_pred_dollar)
    r2_dollar = r2_score(y_true_dollar, y_pred_dollar)
    
    return {
        'log_rmse': rmse_log,
        'log_mae': mae_log,
        'log_r2': r2_log,
        'dollar_rmse': rmse_dollar,
        'dollar_mae': mae_dollar,
        'dollar_r2': r2_dollar,
        'predictions_dollar': y_pred_dollar,
        'actuals_dollar': y_true_dollar
    }

def plot_residuals(y_true_dollar, y_pred_dollar, output_dir: str) -> None:
    """
    Creates a residual plot and a predicted vs actual scatter plot for evaluation.
    """
    os.makedirs(output_dir, exist_ok=True)
    residuals = y_true_dollar - y_pred_dollar
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Limit plots to $800 to avoid outlier scaling issue
    mask = (y_true_dollar < 800) & (y_pred_dollar < 800)
    y_true_f = y_true_dollar[mask]
    y_pred_f = y_pred_dollar[mask]
    res_f = residuals[mask]
    
    # 1. Predicted vs Actual
    sns.scatterplot(x=y_pred_f, y=y_true_f, alpha=0.3, ax=axes[0], color='#2B6CB0')
    axes[0].plot([0, 800], [0, 800], color='red', linestyle='--')
    axes[0].set_title('Predicted vs Actual Prices (Prices < $800)', fontweight='bold')
    axes[0].set_xlabel('Predicted Price ($)')
    axes[0].set_ylabel('Actual Price ($)')
    
    # 2. Residual Plot
    sns.scatterplot(x=y_pred_f, y=res_f, alpha=0.3, ax=axes[1], color='#DD6B20')
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_title('Residuals vs Predicted Prices (Prices < $800)', fontweight='bold')
    axes[1].set_xlabel('Predicted Price ($)')
    axes[1].set_ylabel('Residual Error ($)')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'residuals_and_predictions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved residual plots to {output_path}")

def plot_feature_importance(model, feature_names, output_dir: str) -> None:
    """
    Plots the feature importance of a model (e.g. Random Forest or Gradient Boosting).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute.")
        return
        
    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis', hue='Feature', legend=False)
    plt.title('Feature Importances in Pricing Model', fontweight='bold', pad=15)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    
    output_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance plot to {output_path}")
