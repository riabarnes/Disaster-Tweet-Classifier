#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Step 1: Import Libraries
import pandas as pd
import re
import string
import nltk

# Download stopwords (only runs once)
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Step 2: Load Dataset
df = pd.read_csv("train.csv")  # Make sure train.csv is in the same folder
print("Dataset Shape:", df.shape)
print(df.head())

# Step 3: Preprocessing Function
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = re.sub(r'@\w+|#','', text)  # remove mentions & hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    return " ".join(tokens)

# Apply preprocessing
df['text'] = df['text'].apply(lambda x: clean_text(str(x)))
print("\nSample Cleaned Tweet:", df['text'].iloc[0])

# Step 4: Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['target'], test_size=0.2, random_state=42)

# Step 5: TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train Logistic Regression Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Step 7: Predictions & Evaluation
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_vec)

print("\n✅ Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Test with Custom Tweets
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




