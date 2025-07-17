# --- Standard Libraries ---
import pandas as pd  # For data manipulation
import re  # For regular expressions
import joblib  # For saving and loading models
from bs4 import BeautifulSoup  # For HTML tag removal
# --- NLP and similarity tools ---
from nltk.corpus import stopwords  # Common English stop words
import distance  # For longest common substring
from fuzzywuzzy import fuzz  # For fuzzy string matching

# --- Load model and feature list ---
model = joblib.load("quora_rf_model.pkl")
features = joblib.load("features_list.pkl")

# --- Preprocessing function ---
def custom_preprocess(q):
    q = str(q).lower().strip()
    q = q.replace('%', ' percent').replace('$', ' dollar ').replace('₹', ' rupee ').replace('€', ' euro ').replace('@', ' at ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    contractions = {"can't": "can not", "won't": "will not", "i'm": "i am", "it's": "it is", "don't": "do not", "i've": "i have", "you've": "you have"}
    q = ' '.join([contractions.get(w, w) for w in q.split()])
    q = BeautifulSoup(q, "html.parser").get_text()
    return re.sub(r'\W+', ' ', q).strip()

# --- Feature Functions ---
def common_words(r): return len(set(r['question1'].split()) & set(r['question2'].split()))
def total_words(r): return len(set(r['question1'].split()) | set(r['question2'].split()))

def fetch_token_features(r):
    q1, q2 = r['question1'].split(), r['question2'].split()
    if not q1 or not q2: return [0.0]*8
    STOP = stopwords.words("english")
    q1w, q2w = set([w for w in q1 if w not in STOP]), set([w for w in q2 if w not in STOP])
    q1s, q2s = set([w for w in q1 if w in STOP]), set([w for w in q2 if w in STOP])
    safe_div = 0.0001
    return [
        len(q1w & q2w) / (min(len(q1w), len(q2w)) + safe_div),
        len(q1w & q2w) / (max(len(q1w), len(q2w)) + safe_div),
        len(q1s & q2s) / (min(len(q1s), len(q2s)) + safe_div),
        len(q1s & q2s) / (max(len(q1s), len(q2s)) + safe_div),
        len(set(q1) & set(q2)) / (min(len(q1), len(q2)) + safe_div),
        len(set(q1) & set(q2)) / (max(len(q1), len(q2)) + safe_div),
        int(q1[-1] == q2[-1]), int(q1[0] == q2[0])
    ]

def fetch_length_features(r):
    q1, q2 = r['question1'], r['question2']
    if not q1 or not q2: return [0.0]*3
    l1, l2 = q1.split(), q2.split()
    substrs = list(distance.lcsubstrings(q1, q2))
    return [abs(len(l1)-len(l2)), (len(l1)+len(l2))/2, len(substrs[0])/(min(len(q1),len(q2))+1) if substrs else 0.0]

def fetch_fuzzy_features(r):
    q1, q2 = r['question1'], r['question2']
    return [fuzz.QRatio(q1, q2), fuzz.partial_ratio(q1, q2), fuzz.token_sort_ratio(q1, q2), fuzz.token_set_ratio(q1, q2)]

# --- User Input ---
q1 = input("Enter Question 1: ")
q2 = input("Enter Question 2: ")

# --- Create dataframe and apply preprocessing ---
test_df = pd.DataFrame([[custom_preprocess(q1), custom_preprocess(q2)]], columns=['question1', 'question2'])
test_df['q1_len'] = test_df['question1'].str.len()
test_df['q2_len'] = test_df['question2'].str.len()
test_df['q1_num_words'] = test_df['question1'].apply(lambda x: len(x.split()))
test_df['q2_num_words'] = test_df['question2'].apply(lambda x: len(x.split()))
test_df['word_common'] = test_df.apply(common_words, axis=1)
test_df['word_total'] = test_df.apply(total_words, axis=1)
test_df['word_share'] = round(test_df['word_common'] / (test_df['word_total'] + 1e-6), 2)
# --- Extract token, length, and fuzzy features ---
token_features = test_df.apply(fetch_token_features, axis=1)
length_features = test_df.apply(fetch_length_features, axis=1)
fuzzy_features = test_df.apply(fetch_fuzzy_features, axis=1)
# --- Assign extracted features to DataFrame ---
test_df[['cwc_min','cwc_max','csc_min','csc_max','ctc_min','ctc_max','last_word_eq','first_word_eq']] = pd.DataFrame(token_features.tolist(), index=test_df.index)
test_df[['abs_len_diff','mean_len','longest_substr_ratio']] = pd.DataFrame(length_features.tolist(), index=test_df.index)
test_df[['fuzz_ratio','fuzz_partial_ratio','token_sort_ratio','token_set_ratio']] = pd.DataFrame(fuzzy_features.tolist(), index=test_df.index)
# --- Make prediction ---
X_test = test_df[features]
y_pred = model.predict(X_test)[0]

# --- Output result ---
print("\nPrediction Result:")
print("Question 1:", q1)
print("Question 2:", q2)
print("→ These questions are", "Duplicate" if y_pred == 1 else "Not Duplicate")