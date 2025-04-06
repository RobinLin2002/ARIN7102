import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import re
import joblib
import matplotlib
matplotlib.use('TkAgg')

# 1. Load Data and Preprocessing
# Load the dataset
data = pd.read_csv('Combined Data.csv')

# Drop unnecessary columns
data = data.drop(columns=['Unnamed: 0'])

# Check for missing values
data = data.dropna()

# Clean the text data
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

data['statement'] = data['statement'].apply(clean_text)

# Define features and target
X = data['statement']
y = data['status']

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3. Create a pipeline with TF-IDF and Naive Bayes
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10000)),
    ('nb', MultinomialNB())
])

# 4. Fit the model
pipeline.fit(X_train, y_train)

# 5. Make predictions
y_pred = pipeline.predict(X_test)

# 6. Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# 7. Perform cross-validation with stratified folds
cv = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
print(f"Cross-Validation Accuracy: {cv_scores.mean()} ± {cv_scores.std()}")

# 8. Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
print("Confusion Matrix:")
print(conf_matrix)

labels = ['Anxiety', 'Bipolar', 'Depression', 'Normal', 'Personality \n disorder', 'Stress', 'Suicidal']
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# 9. Analyze misclassifications
errors = pd.DataFrame({
    'Statement': X_test,
    'True Label': y_test,
    'Predicted Label': y_pred
})

# Filter to show only misclassified samples
misclassified = errors[errors['True Label'] != errors['Predicted Label']]
print("\nMisclassified Samples:")
print(misclassified.head(10))

# 10. Sample Sentiment Prediction
sample_statements = [
    "I feel fantastic today!",
    "I'm struggling with anxiety and depression.",
    "Life is going well, and I'm feeling positive.",
    "I'm overwhelmed and don't know how to cope.",
    "Everything seems so hopeless right now."
]

# Predict sentiment for the sample data
predictions = pipeline.predict(sample_statements)

# Print the results
for statement, prediction in zip(sample_statements, predictions):
    print(f"Statement: {statement}\nPredicted Sentiment: {prediction}\n")


# save model
joblib.dump(pipeline, 'bayes result/bayes_pipeline.pkl')

# load model
loaded_pipeline = joblib.load('bayes result/bayes_pipeline.pkl')
