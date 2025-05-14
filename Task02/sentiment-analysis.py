import os
import pandas as pd
import nltk as nl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Download NLTK resources
nl.download('punkt')
nl.download('punkt_tab')  # Added to resolve the punkt_tab error
nl.download('stopwords')
nl.download('wordnet')

# Step 1: Load Reviews
def load_reviews_from_folder(folder_path):
    texts = []
    labels = []

    for label in ['pos', 'neg']:
        folder = os.path.join(folder_path, label)
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(1 if label == 'pos' else 0)

    return pd.DataFrame({'review': texts, 'label': labels})

# Load training data
folder_path = "C:/Users/Ansar-PC/.cache/kagglehub/datasets/pawankumargunjan/imdb-review/versions/3/aclImdb/train"
df = load_reviews_from_folder(folder_path)

# Step 2: Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered = [word for word in tokens if word.isalpha() and word not in stop_words]
    lemmatized = [lemmatizer.lemmatize(word) for word in filtered]
    return ' '.join(lemmatized)

df['clean_review'] = df['review'].apply(preprocess_text)

# Step 3: Feature Engineering (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_review'])
y = df['label']

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Step 7: Predict on New Text
def predict_sentiment(text):
    cleaned = preprocess_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "Positive" if prediction == 1 else "Negative"

# Test the prediction function
sample_text = "This movie was really touching and beautifully made."
print(f"\nSentiment Prediction for sample: {predict_sentiment(sample_text)}")