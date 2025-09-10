"""
visualizations.py

This script generates and saves additional visualizations for:
- Employee rankings (barplot for a selected month)
- Flight risk analysis (barplot of count)
- Model performance (scatter plot of predicted vs actual)
All plots are saved in the 'visualizations' folder.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure output folder exists
os.makedirs('visualizations', exist_ok=True)

# 1. Employee Rankings Visualization
rankings = pd.read_csv('data/rankings.csv')
month = rankings['month'].unique()[0]  # Example: first month
month_rankings = rankings[rankings['month'] == month]
plt.figure(figsize=(8, 4))
sns.barplot(
    data=month_rankings,
    x='employee', y='sentiment_score', hue='type', dodge=True
)
plt.title(f'Employee Rankings for {month}')
plt.ylabel('Sentiment Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/employee_rankings.png')
plt.close()


# 2. Model Performance Visualization (from linear_model.py logic)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load labeled data and compute features as in linear_model.py
messages = pd.read_csv('data/test_labeled.csv')
messages['date'] = pd.to_datetime(messages['date'], errors='coerce')
messages = messages.dropna(subset=['date'])
messages['month'] = messages['date'].dt.to_period('M').astype(str)
messages['msg_length'] = messages['body'].astype(str).apply(len)
messages['word_count'] = messages['body'].astype(str).apply(lambda x: len(x.split()))
agg = messages.groupby(['from', 'month']).agg(
    message_count = ('body', 'count'),
    avg_msg_length = ('msg_length', 'mean'),
    avg_word_count = ('word_count', 'mean'),
).reset_index()
scores = pd.read_csv('data/sentiment_scores.csv')
scores['month'] = scores['month'].astype(str)
data = pd.merge(agg, scores, left_on=['from', 'month'], right_on=['employee', 'month'])
X = data[['message_count', 'avg_msg_length', 'avg_word_count']]
y = data['sentiment_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual Sentiment Score')
plt.ylabel('Predicted Sentiment Score')
plt.title('Model Performance: Actual vs Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.tight_layout()
plt.savefig('visualizations/model_performance.png')
plt.close()

# 3. Model Performance Visualization
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load features and scores for model
monthly_features = pd.read_csv('data/monthly_features.csv') if os.path.exists('data/monthly_features.csv') else None
scores = pd.read_csv('data/sentiment_scores.csv')
if monthly_features is not None:
    data = pd.merge(monthly_features, scores, left_on=['from', 'month'], right_on=['employee', 'month'])
    X = data[['message_count', 'avg_msg_length', 'avg_word_count']]
    y = data['sentiment_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel('Actual Sentiment Score')
    plt.ylabel('Predicted Sentiment Score')
    plt.title('Model Performance: Actual vs Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.tight_layout()
    plt.savefig('visualizations/model_performance.png')
    plt.close()

print("All visualizations saved to the 'visualizations' folder.")
