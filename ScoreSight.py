import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib

def load_config(file_path='config.json'):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def load_data(path):
    data = pd.read_csv(path)
    label_encoder = LabelEncoder()
    data['Name_Encoded'] = label_encoder.fit_transform(data['Name'])
    return data

def plot_heatmap(data, targets):
    plt.figure(figsize=(8, 6))
    sns.heatmap(data[targets].corr(), annot=True, cmap='coolwarm')
    plt.title('Subject Score Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def train_model(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = MultiOutputRegressor(LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    return model

def predict_scores(model, data, inputs, targets, output_path):
    X = data[inputs]
    predictions = model.predict(X)
    predictions_df = pd.DataFrame(predictions, columns=[f'Predicted_{col}' for col in targets])
    results = pd.concat([data[['Name']], predictions_df], axis=1)
    pd.set_option('display.max_columns', None)
    print("Predicted Scores:")
    print(results)
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to '{output_path}'")
    return predictions_df

def plot_predictions(data, predictions_df, targets):
    for subject in targets:
        plt.figure(figsize=(6, 4))
        plt.plot(data['Name'], data[subject], marker='o', label='Actual')
        plt.plot(data['Name'], predictions_df[f'Predicted_{subject}'], marker='x', linestyle='--', label='Predicted')
        plt.title(f'{subject} Scores: Actual vs Predicted')
        plt.xlabel('Student Name')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def main():
    config = load_config()
    data = load_data(config['dataset_path'])

    print("Using configuration settings from config.json")
    plot_heatmap(data, config['target_features'])

    X, y = data[config['input_features']], data[config['target_features']]
    model = train_model(X, y, config['test_size'], config['random_state'])

    joblib.dump(model, config['model_path'])
    print(f"Model saved to '{config['model_path']}'")

    predictions_df = predict_scores(model, data, config['input_features'], config['target_features'], config['output_path'])
    plot_predictions(data, predictions_df, config['target_features'])

if __name__ == "__main__":
    main()

