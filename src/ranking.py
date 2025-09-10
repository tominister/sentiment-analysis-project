"""
ranking.py

This script ranks employees by their monthly sentiment scores.
- Top Three Positive Employees: Highest positive scores per month
- Top Three Negative Employees: Most negative scores per month

Ranking is based on the output of score_calculation.py (monthly_employee_sentiment_scores.csv).

Sorting: First by score (descending for positive, ascending for negative), then alphabetically by employee.
"""
import pandas as pd

INPUT_CSV = "data/sentiment_scores.csv"

# Load the monthly sentiment scores
df = pd.read_csv(INPUT_CSV)

# Ensure correct types
df['month'] = df['month'].astype(str)
df['sentiment_score'] = df['sentiment_score'].astype(int)

# Prepare results
top_positive = []
top_negative = []

for month, group in df.groupby('month'):
    # Top 3 positive: sort by score desc, then employee asc
    pos = group.sort_values(['sentiment_score', 'employee'], ascending=[False, True]).head(3)
    pos['rank'] = [1, 2, 3][:len(pos)]
    pos['type'] = 'Top Positive'
    pos['month'] = month
    top_positive.append(pos)
    # Top 3 negative: sort by score asc, then employee asc
    neg = group.sort_values(['sentiment_score', 'employee'], ascending=[True, True]).head(3)
    neg['rank'] = [1, 2, 3][:len(neg)]
    neg['type'] = 'Top Negative'
    neg['month'] = month
    top_negative.append(neg)

# Combine and format
ranking_df = pd.concat(top_positive + top_negative, ignore_index=True)
ranking_df = ranking_df[['month', 'type', 'rank', 'employee', 'sentiment_score']]

# Save to CSV
table_path = "data/rankings.csv"
ranking_df.to_csv(table_path, index=False)

# Print tables for each month
for month in sorted(df['month'].unique()):
    print(f"\nMonth: {month}")
    print("Top 3 Positive Employees:")
    print(ranking_df[(ranking_df['month'] == month) & (ranking_df['type'] == 'Top Positive')][['rank', 'employee', 'sentiment_score']].to_string(index=False))
    print("Top 3 Negative Employees:")
    print(ranking_df[(ranking_df['month'] == month) & (ranking_df['type'] == 'Top Negative')][['rank', 'employee', 'sentiment_score']].to_string(index=False))

print(f"\nRanking tables saved to {table_path}")

# Discussion:
# Rankings are determined by sorting employees' monthly sentiment scores, as computed in Task 3.
# For each month, the top three with the highest scores are the most positive, and the three with the lowest are the most negative.
# Ties are broken alphabetically by employee email.
