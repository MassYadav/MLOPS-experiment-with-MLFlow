import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

# ==== Dagshub Initialization ====
import dagshub
dagshub.init(repo_owner='MassYadav', repo_name='MLOPS-experiments-with-MLFlow', mlflow=True)

# Explicitly set MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/MassYadav/MLOPS-experiments-with-MLFlow.mlflow")

# ==== Load Dataset ====
wine = load_wine()
X = wine.data
y = wine.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Model parameters
max_depth = 10
n_estimators = 10

# Set MLflow experiment
mlflow.set_experiment('YT-MLOPS-Exp2')

# ==== Start MLflow run ====
with mlflow.start_run():
    # Train model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Predictions & accuracy
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and params
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wine.target_names,
                yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save and log plot
    plt.savefig("Confusion-matrix.png")
    mlflow.log_artifact("Confusion-matrix.png")

    # Log script itself
    mlflow.log_artifact(__file__)

    # Add tags
    mlflow.set_tags({"Author": 'Manish', "Project": "Wine Classification"})

    # Log the model (disable registry to avoid Dagshub unsupported endpoint error)
    import joblib
    joblib.dump(rf, "rf.pkl")
    mlflow.log_artifact("rf.pkl", artifact_path="model")

 


    print(f"Accuracy: {accuracy}")

print("✅ MLflow logging completed and pushed to Dagshub.")
