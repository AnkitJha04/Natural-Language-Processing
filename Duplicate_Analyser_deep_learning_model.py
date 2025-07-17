# --- Import Libraries ---
import numpy as np
import pandas as pd
import re
import nltk
import warnings
import joblib
import gensim
import matplotlib.pyplot as plt

# --- NLP Specific Imports ---
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from gensim.models import Word2Vec

# --- Deep Learning Imports ---
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Bidirectional, Dropout, Dense, Subtract, Multiply, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# --- Evaluation and Utility ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from keras.callbacks import EarlyStopping

# --- NLTK Downloader with Error Handling ---
def safe_nltk_download(package):
    try:
        nltk.data.find(f'corpora/{package}')
    except LookupError:
        nltk.download(package)

safe_nltk_download('stopwords')
safe_nltk_download('wordnet')
warnings.filterwarnings('ignore')

# --- Load Dataset ---
df = pd.read_csv("C:/Users/ankit/OneDrive/Documents/Personal/Repls/Complete code/Natural Language Processing/Duplicate Analyser/Dataset/train.csv")
df = df.dropna(subset=['question1', 'question2']).reset_index(drop=True)

# --- Preprocessing Function ---
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))

def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r'[\d]+', '', text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\W+', ' ', text)
    tokens = text.split()
    words = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens if word not in STOP_WORDS]
    return ' '.join(words)

df['question1_clean'] = df['question1'].apply(preprocess)
df['question2_clean'] = df['question2'].apply(preprocess)

# --- Tokenization and Padding ---
questions = pd.Series(df['question1_clean'].tolist() + df['question2_clean'].tolist())
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)
word_index = tokenizer.word_index

q1_seq = tokenizer.texts_to_sequences(df['question1_clean'])
q2_seq = tokenizer.texts_to_sequences(df['question2_clean'])

# 🔥 Dynamic max length for padding
max_seq_length = max(max(len(seq) for seq in q1_seq), max(len(seq) for seq in q2_seq))
q1_pad = pad_sequences(q1_seq, maxlen=max_seq_length, padding='post')
q2_pad = pad_sequences(q2_seq, maxlen=max_seq_length, padding='post')

# --- Word2Vec Training ---
sentences = [q.split() for q in questions]
w2v_model = Word2Vec(sentences=sentences, vector_size=300, window=5, min_count=5, workers=4)
w2v_model.save("C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/w2v_model.model")

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

# --- Feature Engineering ---
def compute_features(row):
    q1, q2 = row['question1_clean'], row['question2_clean']
    v1 = np.mean([w2v_model.wv[word] for word in q1.split() if word in w2v_model.wv] or [np.zeros(300)], axis=0)
    v2 = np.mean([w2v_model.wv[word] for word in q2.split() if word in w2v_model.wv] or [np.zeros(300)], axis=0)
    cos_sim = cosine_similarity([v1], [v2])[0][0]
    fuzz_ratio = fuzz.ratio(q1, q2)
    fuzz_partial = fuzz.partial_ratio(q1, q2)
    fuzz_token_sort = fuzz.token_sort_ratio(q1, q2)
    fuzz_token_set = fuzz.token_set_ratio(q1, q2)
    return pd.Series([cos_sim, fuzz_ratio, fuzz_partial, fuzz_token_sort, fuzz_token_set])

feature_df = df.apply(compute_features, axis=1)
feature_df.columns = ['cos_sim', 'fuzz_ratio', 'fuzz_partial', 'fuzz_sort', 'fuzz_set']
X_extra = feature_df.values

# --- Train-Test Split ---
X_train_q1, X_test_q1, X_train_q2, X_test_q2, X_train_extra, X_test_extra, y_train, y_test = train_test_split(
    q1_pad, q2_pad, X_extra, df['is_duplicate'].values, test_size=0.2, random_state=42)

# --- Deep Learning Architecture ---
input_q1 = Input(shape=(None,))
input_q2 = Input(shape=(None,))
input_extra = Input(shape=(5,))

embedding_layer = Embedding(input_dim=len(word_index)+1,
                             output_dim=300,
                             weights=[embedding_matrix],
                             input_length=None,
                             trainable=False)

encoded_q1 = embedding_layer(input_q1)
encoded_q2 = embedding_layer(input_q2)

shared_bilstm = Bidirectional(LSTM(64, return_sequences=False))
q1_vec = shared_bilstm(encoded_q1)
q2_vec = shared_bilstm(encoded_q2)

subtracted = Subtract()([q1_vec, q2_vec])
multiplied = Multiply()([q1_vec, q2_vec])
merged = Concatenate()([subtracted, multiplied, input_extra])

x = Dense(64, activation='relu')(merged)
x = Dropout(0.4)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[input_q1, input_q2, input_extra], outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# --- Train the Model ---
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit([X_train_q1, X_train_q2, X_train_extra], y_train,
          batch_size=128, epochs=10, validation_split=0.1, callbacks=[early_stop])

# --- Evaluate Model ---
y_pred = (model.predict([X_test_q1, X_test_q2, X_test_extra]) > 0.5).astype("int32")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# --- Save Model & Tokenizer ---
model.save("C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/word2vec_dl_model.h5")
joblib.dump(tokenizer, "C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/tokenizer.pkl")