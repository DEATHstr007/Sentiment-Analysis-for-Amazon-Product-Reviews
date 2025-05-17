from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Sample dataset
df = pd.DataFrame({
    "text": [
        "The Strawberry Twizzlers are my guilty pleasure - yummy. "
        "Six pounds will be around for a while with my son and I."
    ]
})

# Compute sentiment scores
df["sentiment_score"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Optional: Convert score to a label (Positive, Negative, Neutral)
df["sentiment_label"] = df["sentiment_score"].apply(
    lambda score: "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
)

print(df)
