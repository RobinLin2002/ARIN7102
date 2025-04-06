import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
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
# # show distribution
# plt.figure(figsize=(12, 7))
# sns.countplot(data=df, x='status')
# plt.title('Distribution of Status')
# plt.show()

# # average characters and length
# # characters length
# df['statment_length']=df['statement'].apply(lambda x:len(x))
# # words length
# df['num_of_words']=df['statement'].apply(lambda x:len(x.split()))
# status_analysis = df.groupby('status').agg({
#     'statment_length': ['mean', 'median', 'std'],
#     'num_of_words': ['mean', 'median', 'std']
# })
# status_analysis['statment_length']['mean'].plot(kind='bar',color='skyblue')
# plt.title('Avarage of characters per statment')
# plt.show()
#
# status_analysis['num_of_words']['mean'].plot(kind='bar',color='purple',alpha=0.4)
# plt.title('Avarage of words per statment')
# plt.show()

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

# Train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
xgb_model.fit(X_train_tfidf, y_train)

# save model
with open('xgboost result/xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# Predict and evaluate
preds = xgb_model.predict(X_val_tfidf)
acc = accuracy_score(y_val, preds)
print(f"XGBoost Accuracy: {acc:.4f}")
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
# with open('xgb_model.pkl', 'rb') as f:
#     xgb_model_loaded = pickle.load(f)


