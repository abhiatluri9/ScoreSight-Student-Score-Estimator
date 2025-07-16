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

# Load and encode data
def load_data(file_path):
    data = pd.read_csv(file_path)
    label_encoder = LabelEncoder()
    data['Name_Encoded'] = label_encoder.fit_transform(data['Name'])
    return data

# Plot heatmap
def plot_heatmap(data, targets):
    plt.figure(figsize=(8, 6))
    sns.heatmap(data[targets].corr(), annot=True, cmap='coolwarm')
    plt.title('Subject Score Correlation Heatmap')
    plt.tight_layout()
    plt.show()

# Split features and targets
def split_data(data, input_features, target_columns):
    X = data[input_features]
    y = data[target_columns]
    return X, y

# Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = MultiOutputRegressor(LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nMean Squared Error (average across subjects): {mse:.2f}")
    return model

# Predict and display
def predict_scores(model, data, input_features, target_columns):
    X = data[input_features]
    predictions = model.predict(X)
    predictions_df = pd.DataFrame(predictions, columns=[f'Predicted_{col}' for col in target_columns])
    pd.set_option('display.max_columns', None)
    results = pd.concat([data[['Name']], predictions_df], axis=1)
    print("\nPredicted Scores:")
    print(results)
    results.to_csv('predicted_scores.csv', index=False)
    print("Predictions saved to 'predicted_scores.csv'")
    return predictions_df

# Plot predictions
def plot_predictions(data, predictions_df, target_columns):
    for subject in target_columns:
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

# Main function
def main():
    file_path = 'student_scores.csv'
    input_features = ['Study_Math', 'Study_Science', 'Study_English', 'Study_Social',
                      'Attendance', 'Homework_Completion', 'Participation']
    target_columns = ['Math', 'Science', 'English', 'social']

    print("Loading student data...")
    data = load_data(file_path)

    print("Generating correlation heatmap...")
    plot_heatmap(data, target_columns)

    print("Training prediction model...")
    X, y = split_data(data, input_features, target_columns)
    model = train_model(X, y)

    joblib.dump(model, 'scoresight_model.pkl')
    print("Model saved to 'scoresight_model.pkl'")

    print("Making predictions...")
    predictions_df = predict_scores(model, data, input_features, target_columns)

    print("Plotting actual vs predicted scores...")
    plot_predictions(data, predictions_df, target_columns)

if __name__ == "__main__":
    main()
