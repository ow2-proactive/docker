import pathlib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

def simulate():
    # Create simulated regression data
    X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Set the MLflow tracking URI (replace with your server address)
mlflow.set_tracking_uri("http://localhost:5000")
print("Tracking URI set to:", mlflow.get_tracking_uri())

# Create and set the experiment
mlflow.create_experiment('my_first_experiment')
mlflow.set_experiment('my_first_experiment')

# Generate the data
X_train, X_test, y_train, y_test = simulate()

# Define models to train
models = {
    "Decision Tree": DecisionTreeRegressor(max_depth=5),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5),
    "Support Vector Machine (SVM)": SVR(C=1.0, kernel='linear', epsilon=0.1),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    "Ridge Regression": Ridge(alpha=1.0)
}

# Train and log individual models
for model_name, model in models.items():
    with mlflow.start_run():
        print(model_name)
        
        # Create and save feature scatter plot
        fig, ax = plt.subplots()
        ax.scatter(X_train[0,:], X_train[1,:])
        ax.set_title("Feature Scatter Plot", fontsize=14)
        plt.tight_layout()
        save_path = pathlib.Path("./tmp/scatter_plot.png")
        fig.savefig(save_path)
        
        # Train the model
        model.fit(X_train, y_train)
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate Mean Squared Error (MSE) and R²
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(model, model_name, signature=signature)
        
        # Log hyperparameters and metrics
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("parameters", model.get_params())
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.log_artifact("./tmp/scatter_plot.png")

# Compare all models in a single run
model_comparison = {
    "Decision Tree": {"mse": 0, "r2": 0},
    "Random Forest": {"mse": 0, "r2": 0},
    "Support Vector Machine (SVM)": {"mse": 0, "r2": 0},
    "K-Nearest Neighbors": {"mse": 0, "r2": 0},
    "Ridge Regression": {"mse": 0, "r2": 0}
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    model_comparison[model_name]["mse"] = mse
    model_comparison[model_name]["r2"] = r2

with mlflow.start_run():
    # Convert to DataFrame for visualization
    comparison_df = pd.DataFrame(model_comparison).T
    comparison_df.plot(kind="bar", figsize=(10, 6), title="Model Comparison: MSE and R²")
    plt.ylabel("Value")
    plt.savefig("./tmp/model_comparison.png")
    mlflow.log_artifact("./tmp/model_comparison.png")