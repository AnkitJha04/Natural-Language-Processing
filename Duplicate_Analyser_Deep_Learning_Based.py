# --- Import Libraries ---
import numpy as np
import re
import pickle
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz

# --- NLTK Downloads ---
nltk.download('stopwords')
nltk.download('wordnet')

# --- NLP Setup ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- Load Resources ---
tokenizer_path = "C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/tokenizer.pkl"
w2v_model_path = "C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/w2v_model.model"
model_path = "C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/word2vec_dl_model.h5"

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

w2v_model = Word2Vec.load(w2v_model_path)
model = load_model(model_path)

# --- Preprocessing ---
def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r'[%$@₹€]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = BeautifulSoup(text, "html.parser").get_text()
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

# --- Sentence Vector (Average Word2Vec) ---
def sentence_vector(text):
    words = text.split()
    vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(300)

# --- Extra Features ---
def get_extra_features(q1, q2):
    v1 = sentence_vector(q1)
    v2 = sentence_vector(q2)
    cos_sim = cosine_similarity([v1], [v2])[0][0]
    fuzz_ratio = fuzz.ratio(q1, q2)
    fuzz_partial = fuzz.partial_ratio(q1, q2)
    fuzz_token_sort = fuzz.token_sort_ratio(q1, q2)
    fuzz_token_set = fuzz.token_set_ratio(q1, q2)
    return np.array([[cos_sim, fuzz_ratio, fuzz_partial, fuzz_token_sort, fuzz_token_set]])

# --- User Input ---
q1 = input("Enter Question 1: ")
q2 = input("Enter Question 2: ")

# --- Preprocess ---
q1_clean = preprocess(q1)
q2_clean = preprocess(q2)

# --- Tokenize ---
q1_seq = tokenizer.texts_to_sequences([q1_clean])
q2_seq = tokenizer.texts_to_sequences([q2_clean])

# --- Dynamic Padding ---
max_len = max(len(q1_seq[0]), len(q2_seq[0]))  # Adjust dynamically
q1_pad = pad_sequences(q1_seq, maxlen=max_len, padding='post')
q2_pad = pad_sequences(q2_seq, maxlen=max_len, padding='post')

# --- Extra Features ---
extra_feat = get_extra_features(q1_clean, q2_clean)

# --- Predict ---
pred = model.predict([q1_pad, q2_pad, extra_feat])[0][0]
print("\nDuplicate Probability: {:.2f}%".format(pred * 100))
if pred > 0.5:
    print("✅ The questions are likely DUPLICATES.")
else:
    print("❌ The questions are NOT duplicates.")