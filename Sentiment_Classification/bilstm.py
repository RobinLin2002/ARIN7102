import numpy as np
import pandas as pd
import pickle
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Load dataset
df = pd.read_csv('Combined Data.csv')
df.dropna(subset=['statement'], inplace=True)
df.reset_index(drop=True, inplace=True)

# 2. Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['status'])

# 3. Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    df['statement'], df['label'], test_size=0.1, stratify=df['label'], random_state=42
)

# 4. TF-IDF to get vocabulary
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 3),
    stop_words='english',
    min_df=1,
    max_df=0.8
)
vectorizer.fit(X_train)

# Save vectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# 5. Build tokenizer
tokenizer = Tokenizer(num_words=30000, oov_token="<oov>")
tokenizer.word_index = vectorizer.vocabulary_
tokenizer.index_word = {v: k for k, v in tokenizer.word_index.items()}
tokenizer.fit_on_texts(X_train)

# Save tokenizer
with open('tokenizer_from_tfidf.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# 6. Convert to sequences
MAX_SEQUENCE_LENGTH = 100
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

y_train = np.array(y_train)
y_val = np.array(y_val)

# 7. Build Bidirectional LSTM model
model = Sequential([
    Embedding(input_dim=30000, output_dim=128, input_length=MAX_SEQUENCE_LENGTH),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 8. Train model
model.fit(X_train_pad, y_train, epochs=10, batch_size=64, validation_data=(X_val_pad, y_val))

# 9. Save model
model.save('lstm_result/bilstm_model_final.h5')

# 10. Predict and evaluate
y_pred_probs = model.predict(X_val_pad)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_val, y_pred)
print(f"Bidirectional LSTM Accuracy: {acc:.4f}")
print(classification_report(y_val, y_pred, target_names=le.classes_))

# 11. Confusion Matrix
labels = ['Anxiety', 'Bipolar', 'Depression', 'Normal', 'Personality \n disorder', 'Stress', 'Suicidal']
cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - BiLSTM')
plt.tight_layout()
plt.show()