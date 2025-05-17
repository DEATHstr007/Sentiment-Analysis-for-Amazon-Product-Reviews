import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed data
df = pd.read_csv("amazon_reviews_processed.csv")

# 1. Add sentiment label column based on score
def label_sentiment(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

df["sentiment_label"] = df["sentiment_score"].apply(label_sentiment)

# 2. Pie Chart
plt.figure(figsize=(6, 6))
df["sentiment_label"].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, colors=["lightgreen", "lightblue", "salmon"])
plt.title("Sentiment Distribution")
plt.ylabel("")
plt.show()

# 3. Histogram
plt.figure(figsize=(8, 5))
sns.histplot(df["sentiment_score"], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Sentiment Scores")
plt.xlabel("Sentiment Score")
plt.ylabel("Number of Reviews")
plt.show()

# 4. (Optional) Average sentiment by product rating (if you have 'Score' column)
if "Score" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Score", y="sentiment_score", data=df)
    plt.title("Average Sentiment by Amazon Rating")
    plt.xlabel("Amazon Rating")
    plt.ylabel("Average Sentiment Score")
    plt.show()
