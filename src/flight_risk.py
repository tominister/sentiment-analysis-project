"""
flight_risk.py

This script identifies employees at risk of leaving (flight risk) based on their message sentiment.

Definition:
- An employee is flagged as flight risk if they have sent 4 or more negative messages in any rolling 30-day window (regardless of score).
- The 30-day window is rolling, not restricted to calendar months.

Method:
- For each employee, filter their messages to only negative ones.
- For each negative message, count the number of negative messages in the previous 29 days (including the current one).
- If any window has 4 or more, flag the employee.

The output is a list of unique employee emails at risk.
"""
import pandas as pd

INPUT_CSV = "data/test_labeled.csv"

# Load data
df = pd.read_csv(INPUT_CSV)
if 'date' not in df.columns or 'from' not in df.columns or 'sentiment' not in df.columns:
    raise ValueError("Input CSV must contain 'date', 'from', and 'sentiment' columns.")

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

# Only negative messages
neg_df = df[df['sentiment'] == 'Negative'].copy()

# Sort by employee and date
neg_df = neg_df.sort_values(['from', 'date'])

flight_risk_employees = set()

for employee, group in neg_df.groupby('from'):
    dates = group['date'].sort_values().reset_index(drop=True)
    for i in range(len(dates)):
        # Count how many negative messages in the 30-day window ending at this date
        window_start = dates[i] - pd.Timedelta(days=29)
        count = ((dates >= window_start) & (dates <= dates[i])).sum()
        if count >= 4:
            flight_risk_employees.add(employee)
            break  # Only need to flag once per employee

# Output results
flight_risk_list = sorted(flight_risk_employees)
output_path = "data/flight_risk_employees.csv"
pd.DataFrame({'employee': flight_risk_list}).to_csv(output_path, index=False)

print(f"Flight risk employees saved to {output_path}")
print("\nList of employees at risk:")
for emp in flight_risk_list:
    print(emp)

# Discussion:
# This script robustly checks for any rolling 30-day window with 4+ negative messages per employee.
# The result is a list of unique employees who meet the flight risk criteria at least once.
