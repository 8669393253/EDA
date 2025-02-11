import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset (Use your own dataset or collect tweets using SNScrape)
df = pd.read_csv("https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv")
df = df[['tweet', 'label']]  # Selecting required columns
df.columns = ['text', 'sentiment']

# Mapping sentiment values (0: Negative, 1: Positive)
df['sentiment'] = df['sentiment'].map({0: 'negative', 1: 'positive'})

# Data Cleaning
def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove @mentions
    text = re.sub(r'#', '', text)  # Remove hashtags
    text = re.sub(r'RT[\s]+', '', text)  # Remove retweets
    text = re.sub(r'https?:\/\/\S+', '', text)  # Remove links
    text = re.sub(r'[^a-zA-Z ]', '', text)  # Remove special characters
    return text.lower().strip()

df['cleaned_text'] = df['text'].apply(clean_text)

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(6,4))
sns.countplot(x=df['sentiment'], palette='viridis')
plt.title("Sentiment Distribution")
plt.show()

# WordCloud for Positive and Negative Sentiments
positive_words = ' '.join(df[df['sentiment'] == 'positive']['cleaned_text'])
negative_words = ' '.join(df[df['sentiment'] == 'negative']['cleaned_text'])

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Positive Sentiment WordCloud")
plt.imshow(WordCloud(width=500, height=300).generate(positive_words))
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Negative Sentiment WordCloud")
plt.imshow(WordCloud(width=500, height=300, colormap='Reds').generate(negative_words))
plt.axis('off')
plt.show()

# Sentiment Analysis using TextBlob
def get_sentiment(text):
    return "positive" if TextBlob(text).sentiment.polarity > 0 else "negative"
df['textblob_sentiment'] = df['cleaned_text'].apply(get_sentiment)

# Train Machine Learning Model
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
