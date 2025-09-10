import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    file_path = os.path.join("data", "test_labeled.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Expected file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

def basic_info(df):
    print("\n--- Basic Data Info ---")
    print(df.info())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Sample Rows ---")
    print(df.head())

def sentiment_distribution(df):
    print("\n--- Sentiment Label Distribution ---")
    print(df['sentiment'].value_counts())

    sns.countplot(data=df, x='sentiment', palette='Set2', legend=False)
    plt.title("Sentiment Label Distribution")
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(os.path.join("visualizations", "sentiment_distribution.png"))
    plt.clf()

def sentiment_over_time(df):
    if 'date' not in df.columns:
        print("No 'date' column to analyze time trends.")
        return

    df['month'] = df['date'].dt.to_period('M')
    monthly_counts = df.groupby(['month', 'sentiment']).size().unstack().fillna(0)

    monthly_counts.plot(kind='bar', stacked=True, figsize=(12,6), colormap='Set3')
    plt.title("Sentiment Trend Over Time")
    plt.xlabel("Month")
    plt.ylabel("Number of Messages")
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(os.path.join("visualizations", "sentiment_over_time.png"))
    plt.clf()

def top_employees_by_activity(df, top_n=10):
    # Use 'from' column as employee id if 'employee_id' is missing
    id_col = 'employee_id' if 'employee_id' in df.columns else 'from'
    if id_col not in df.columns:
        print("No suitable column found for employee activity plot. Skipping.")
        return
    activity_counts = df[id_col].value_counts().head(top_n)
    activity_counts.plot(kind='bar', color='skyblue')
    plt.title(f"Top {top_n} Most Active Employees")
    plt.xlabel("Employee ID (email)")
    plt.ylabel("Message Count")
    plt.tight_layout()
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(os.path.join("visualizations", "top_employees.png"))
    plt.clf()

def main():
    os.makedirs("visualizations", exist_ok=True)
    df = load_data()

    basic_info(df)
    sentiment_distribution(df)
    sentiment_over_time(df)
    top_employees_by_activity(df)

    print("\nEDA complete. Visualizations saved to visualizations/")

if __name__ == "__main__":
    main()
