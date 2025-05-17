import pandas as pd
import spacy
from textblob import TextBlob
import re
from tqdm import tqdm  


nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    if pd.isna(text) or text.strip() == "":  
        return ""
    
    text = text.lower()  
    text = re.sub(r'\W+', ' ', text)  
    text = re.sub(r'\d+', '', text)  
    
    
    if len(text.split()) < 3:
        return text
    
    doc = nlp(text)  
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    
    return " ".join(tokens)


def get_sentiment(text):
    if text.strip() == "":  # If empty, return neutral
        return 0.0
    
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Score between -1 (negative) and +1 (positive)

# Load dataset (Ensure the file exists)
df = pd.read_csv("amazon_reviews.csv", encoding="utf-8")

# Verify column names
print("Columns in dataset:", df.columns)

# Check if "Text" column exists
if "Text" not in df.columns:
    raise KeyError("Error: Column 'Text' not found in dataset. Check the file structure.")

# Apply text preprocessing with a progress bar
tqdm.pandas()
df["cleaned_review"] = df["Text"].progress_apply(preprocess_text)


df["sentiment_score"] = df["cleaned_review"].progress_apply(get_sentiment)


df.to_csv("amazon_reviews_processed.csv", index=False)

print("\nProcessing complete! Sentiment analysis saved to 'amazon_reviews_processed.csv'.")
print(df[["Text", "cleaned_review", "sentiment_score"]].head())
