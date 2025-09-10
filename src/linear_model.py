"""
linear_model.py

This script builds a linear regression model to predict monthly sentiment scores for employees based on features such as message frequency, average message length, and word count.

Steps:
1. Feature engineering from the labeled message data.
2. Aggregate features and sentiment scores per employee per month.
3. Split into train/test sets.
4. Fit a linear regression model.
5. Evaluate and interpret results.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load labeled data
messages = pd.read_csv('data/test_labeled.csv')
messages['date'] = pd.to_datetime(messages['date'], errors='coerce')
messages = messages.dropna(subset=['date'])
messages['month'] = messages['date'].dt.to_period('M').astype(str)

# Feature engineering
messages['msg_length'] = messages['body'].astype(str).apply(len)
messages['word_count'] = messages['body'].astype(str).apply(lambda x: len(x.split()))

# Aggregate features per employee per month
agg = messages.groupby(['from', 'month']).agg(
    message_count = ('body', 'count'),
    avg_msg_length = ('msg_length', 'mean'),
    avg_word_count = ('word_count', 'mean'),
).reset_index()


# Load sentiment scores
scores = pd.read_csv('data/sentiment_scores.csv')
scores['month'] = scores['month'].astype(str)

# Merge features and target
data = pd.merge(agg, scores, left_on=['from', 'month'], right_on=['employee', 'month'])

# Prepare features and target
X = data[['message_count', 'avg_msg_length', 'avg_word_count']]
y = data['sentiment_score']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Interpretation
print("\nFeature importances (coefficients):")
for name, coef in zip(['message_count', 'avg_msg_length', 'avg_word_count'], model.coef_):
    print(f"{name}: {coef:.3f}")

print("\nInterpretation:")
print("The coefficients indicate the expected change in sentiment score for a unit change in each feature, holding others constant.")
print("A positive coefficient means higher values of that feature are associated with higher sentiment scores.")
print("Model performance is measured by MSE and R^2. Higher R^2 indicates better fit.")
