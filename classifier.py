#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import re
import string
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

df = pd.read_csv("train.csv") 
print("Dataset Shape:", df.shape)
print(df.head())

def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = re.sub(r'@\w+|#','', text)  # remove mentions & hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    return " ".join(tokens)

df['text'] = df['text'].apply(lambda x: clean_text(str(x)))
print("\nSample Cleaned Tweet:", df['text'].iloc[0])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['target'], test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_vec)

print("\n✅ Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

sample_tweets = [
    "Massive earthquake shakes city center!",
    "Just finished watching a great movie, loved it!",
    "Floods destroyed hundreds of homes",
    "Happy birthday to my best friend!"
]

sample_tweets_clean = [clean_text(t) for t in sample_tweets]
sample_vec = vectorizer.transform(sample_tweets_clean)
predictions = model.predict(sample_vec)

print("\n🔮 Sample Predictions:")
for tweet, pred in zip(sample_tweets, predictions):
    print(f"{tweet} --> {'Disaster' if pred==1 else 'Not Disaster'}")


# In[ ]:




