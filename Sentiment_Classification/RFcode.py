import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import matplotlib
matplotlib.use('TkAgg')

# Load dataset
df = pd.read_csv('Combined Data.csv')
df.dropna(subset=['statement'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['status'])

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    df['statement'], df['label'], test_size=0.1, stratify=df['label'], random_state=42
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    max_features=30000,            # increase to 30k
    ngram_range=(1, 3),            # use unigrams, bigrams, trigrams
    stop_words='english',
    min_df=1,                      # remove very rare words
    max_df=0.8                     # remove overly common words
)
# save TF-IDF
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# # load TF-IDF
# loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')
# tfidf_matrix = loaded_vectorizer.transform(new_documents)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Train RandomForest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_tfidf, y_train)

# save model
with open('RF result/randomforest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Predict and evaluate
preds = rf_model.predict(X_val_tfidf)
acc = accuracy_score(y_val, preds)
print(f"RandomForest Accuracy: {acc:.4f}")
print(classification_report(y_val, preds, target_names=le.classes_))

# confusion_matrix
labels = ['Anxiety', 'Bipolar', 'Depression', 'Normal', 'Personality \n disorder', 'Stress', 'Suicidal']
cm = confusion_matrix(y_val, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# # if you want to load model
# with open('randomforest_model.pkl', 'rb') as f:
#     rf_model_loaded = pickle.load(f)
