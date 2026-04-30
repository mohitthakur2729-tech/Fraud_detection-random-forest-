import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATASET_PATH = "dataset/upi_fraud_dataset.csv"
SCALER_PATH = "filesuse/scaler.pkl"
MODEL_PATH = "filesuse/random_forest_model.pkl"


def main():
    dataset = pd.read_csv(DATASET_PATH, index_col=0)
    x = dataset.iloc[:, :10].values
    y = dataset.iloc[:, 10].values

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.15,
        random_state=0,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=0,
        class_weight="balanced",
        min_samples_leaf=2,
    )
    model.fit(x_train_scaled, y_train)

    predictions = model.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(model, MODEL_PATH)

    print(f"Saved scaler to {SCALER_PATH}")
    print(f"Saved Random Forest model to {MODEL_PATH}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()
