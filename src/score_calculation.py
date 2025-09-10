"""
score_calculation.py

This script computes a monthly sentiment score for each employee (using the 'from' column as employee ID) based on their messages.
- Positive Message: +1
- Negative Message: -1
- Neutral Message: 0

Grouping and aggregation method:
- The script converts the 'date' column to datetime and extracts the month as a period (YYYY-MM).
- Each message is assigned a score based on its sentiment.
- The script groups by employee and month, summing the scores for each group.
- The score resets at the start of each new month (i.e., aggregation is per month, not cumulative across months).

The resulting CSV contains columns: employee, month, sentiment_score.
"""
import pandas as pd
import os

def compute_monthly_sentiment_scores(input_path="data/test_labeled.csv", output_path="data/sentiment_scores.csv"):
    # Load data
    df = pd.read_csv(input_path)
    if 'date' not in df.columns or 'from' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("Input CSV must contain 'date', 'from', and 'sentiment' columns.")
    
    # Convert date to datetime and extract month
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['month'] = df['date'].dt.to_period('M')

    # Map sentiment to score
    sentiment_map = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
    df['score'] = df['sentiment'].map(sentiment_map)

    # Group by employee and month, then sum scores
    monthly_scores = df.groupby(['from', 'month'])['score'].sum().reset_index()
    monthly_scores = monthly_scores.rename(columns={'from': 'employee', 'month': 'month', 'score': 'sentiment_score'})

    # Save results
    monthly_scores.to_csv(output_path, index=False)
    print(f"Monthly sentiment scores saved to {output_path}")

if __name__ == "__main__":
    compute_monthly_sentiment_scores()
