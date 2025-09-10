import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk



def get_sentiment_label(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)['compound']
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

def main():
    df = pd.read_csv("data/test.csv")  # Adjust path if needed
    # Use 'body' column for sentiment if 'message' does not exist
    text_col = 'message' if 'message' in df.columns else 'body'
    if text_col not in df.columns:
        raise ValueError(f"Neither 'message' nor 'body' column found in test.csv. Columns: {df.columns.tolist()}")
    df['sentiment'] = df[text_col].apply(get_sentiment_label)
    # Save to data/test_labeled.csv as printed in the message
    import os
    os.makedirs('data', exist_ok=True)
    df.to_csv("data/test_labeled.csv", index=False)
    print("Sentiment labeling complete. Output saved to data/test_labeled.csv")

if __name__ == "__main__":
    main()
