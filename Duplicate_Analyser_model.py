# --- Importing Necessary Libraries ---
# Standard Libraries
import numpy as np  # For numerical operations like array manipulations
import pandas as pd  # For structured data manipulation via DataFrames
import re  # For regular expression operations (used in text cleaning)
import warnings  # To suppress unnecessary warning messages
# Text Cleaning
from bs4 import BeautifulSoup  # To remove HTML/XML tags from text data
# Natural Language Processing Tools
import nltk  # Natural language processing toolkit
from nltk.corpus import stopwords  # Standard stopword list for filtering
from nltk.stem import WordNetLemmatizer  # For reducing words to their base form
# Text Similarity Libraries
import distance  # To compute longest common substrings
from fuzzywuzzy import fuzz  # For fuzzy matching techniques like token sort/set ratio
# Machine Learning Tools
from sklearn.model_selection import train_test_split  # For splitting dataset into train/test sets
from xgboost import XGBClassifier  # Optimized gradient boosting classifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # Evaluation metrics
from sklearn.utils import resample  # For resampling to balance the dataset
import joblib  # To save and load trained models or feature lists

# --- Load and Clean the Dataset ---
df = pd.read_csv("C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/Quora data/train.csv") # Load Quora duplicate question dataset
df = df.dropna(subset=['question1', 'question2']).reset_index(drop=True) # Remove rows with missing questions
# Download NLTK resources
nltk.download('stopwords')  # For removing stop words
nltk.download('wordnet')  # For doing lemmatization using wordnt lexical dictionary
warnings.filterwarnings("ignore")  # Suppress all warnings during execution 

'''(EDA STEPS TO ANALYSE AND UNDERSTAND THE DATA)
print(new_df.shape)  # Shape of the data set provided 
print(new_df.head(10))  # First 10 data of the data set provided 
print(new_df.info())  # Info on the data set provided 
print(new_df.isnull().sum())  # Checking if there is any discrepency like missing data rows in the data set provided 
print(new_df.duplicated().sum())  # Checking if there is any discrepency like duplicate data rows in the data set provided 
print(new_df['is_duplicate'].value_counts()) # Checking if there is any discrepency like unbalanced data rows in the data set provided
print((new_df['is_duplicate'].value_counts()/new_df['is_duplicate'].count())*100) 
qid = pd.Series(new_df['qid1'].tolist() + new_df['qid2'].tolist())
print('Number of unique questions',np.unique(qid).shape[0])
x = qid.value_counts()>1
print('Number of questions getting repeated',x[x].shape[0])'''

# --- Balancing the Dataset ---
majority = df[df['is_duplicate'] == 0]  # Separate majority and minority classes (0 = not duplicate, 1 = duplicate)
minority = df[df['is_duplicate'] == 1]  # Separate majority and minority classes (0 = not duplicate, 1 = duplicate)
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)  # Upsample the minority class to match the majority class size
df_balanced = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True) # Combine and shuffle to form the balanced dataset

# --- Text Preprocessing Function ---
lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer

def preprocess(q): # Function to Clean and lemmatize input question text
    q = str(q).lower().strip()  # Lowercase and strip spaces
    q = re.sub(r'[%$@₹€]', lambda m: {'%':' percent', '$':' dollar ', '₹':' rupee ', '€':' euro ', '@':' at '}[m.group()], q)  # Replacing multiple characters using RegEx
    q = re.sub(r'(\d+)000000000', r'\1b', q)  # Convert billion
    q = re.sub(r'(\d+)000000', r'\1m', q)  # Convert million
    q = re.sub(r'(\d+)000', r'\1k', q)  # Convert thousand
    q = re.sub(r'[math]', '', q)  # Remove '[math]' markup
    contractions = {"can't":"can not", "won't":"will not", "i'm":"i am", "it's":"it is", "don't":"do not", "i've":"i have", "you've":"you have"}  # Defining the contraction we're going to deal with
    q = ' '.join([contractions.get(word, word) for word in q.split()])  # Expand contractions
    q = BeautifulSoup(q, "html.parser").get_text()  # Strip HTML
    q = re.sub(r'\W+', ' ', q).strip()  # Remove special characters
    q = ' '.join([lemmatizer.lemmatize(word) for word in q.split()])  # Lemmatize each word
    return q

# --- Feature Engineering Functions ---
STOP_WORDS = set(stopwords.words("english"))  # English stopwords

def common_words(row):  # Function to Count the number of shared words
    w1 = set(row['question1'].split())
    w2 = set(row['question2'].split())
    return len(w1 & w2)

def total_words(row):  # Function to Count total unique words in both questions
    w1 = set(row['question1'].split())
    w2 = set(row['question2'].split())
    return len(w1) + len(w2)

def fetch_token_features(row): # Function to Extract token-level overlap and positional features
    q1, q2 = row['question1'], row['question2']
    tf = [0.0]*8
    q1_tokens, q2_tokens = q1.split(), q2.split()
    if not q1_tokens or not q2_tokens: return tf
    q1_words = set([w for w in q1_tokens if w not in STOP_WORDS])
    q2_words = set([w for w in q2_tokens if w not in STOP_WORDS])
    q1_stops = set([w for w in q1_tokens if w in STOP_WORDS])
    q2_stops = set([w for w in q2_tokens if w in STOP_WORDS])
    tf[0] = len(q1_words & q2_words) / (min(len(q1_words), len(q2_words)) + 1e-6)
    tf[1] = len(q1_words & q2_words) / (max(len(q1_words), len(q2_words)) + 1e-6)
    tf[2] = len(q1_stops & q2_stops) / (min(len(q1_stops), len(q2_stops)) + 1e-6)
    tf[3] = len(q1_stops & q2_stops) / (max(len(q1_stops), len(q2_stops)) + 1e-6)
    tf[4] = len(set(q1_tokens) & set(q2_tokens)) / (min(len(q1_tokens), len(q2_tokens)) + 1e-6)
    tf[5] = len(set(q1_tokens) & set(q2_tokens)) / (max(len(q1_tokens), len(q2_tokens)) + 1e-6)
    tf[6] = int(q1_tokens[-1] == q2_tokens[-1])  # Last word match
    tf[7] = int(q1_tokens[0] == q2_tokens[0])  # First word match
    return tf

def fetch_length_features(row): # Function to Capture differences in question lengths and substrings
    q1, q2 = row['question1'], row['question2']
    abs_diff = abs(len(q1.split()) - len(q2.split()))
    mean_len = (len(q1.split()) + len(q2.split())) / 2
    substrs = list(distance.lcsubstrings(q1, q2))
    longest_substr_ratio = len(substrs[0]) / (min(len(q1), len(q2)) + 1) if substrs else 0.0
    return [abs_diff, mean_len, longest_substr_ratio]

def fetch_fuzzy_features(row): # Function to Generate fuzzy string similarity metrics
    q1, q2 = row['question1'], row['question2']
    return [fuzz.QRatio(q1, q2), fuzz.partial_ratio(q1, q2), fuzz.token_sort_ratio(q1, q2), fuzz.token_set_ratio(q1, q2)]

# --- Preprocessing & Feature Extraction on Dataset ---

# Apply preprocessing to both questions
df_balanced['question1'] = df_balanced['question1'].apply(preprocess)
df_balanced['question2'] = df_balanced['question2'].apply(preprocess)
# Basic statistical features
df_balanced['q1_len'] = df_balanced['question1'].str.len()
df_balanced['q2_len'] = df_balanced['question2'].str.len()
df_balanced['q1_num_words'] = df_balanced['question1'].apply(lambda x: len(x.split()))
df_balanced['q2_num_words'] = df_balanced['question2'].apply(lambda x: len(x.split()))
df_balanced['word_common'] = df_balanced.apply(common_words, axis=1)
df_balanced['word_total'] = df_balanced.apply(total_words, axis=1)
df_balanced['word_share'] = round(df_balanced['word_common'] / (df_balanced['word_total'] + 1e-6), 2)
# Extract token, length, and fuzzy features
token_features = df_balanced.apply(fetch_token_features, axis=1)
length_features = df_balanced.apply(fetch_length_features, axis=1)
fuzzy_features = df_balanced.apply(fetch_fuzzy_features, axis=1)
# Append extracted features to dataframe
for i, name in enumerate(['cwc_min','cwc_max','csc_min','csc_max','ctc_min','ctc_max','last_word_eq','first_word_eq']):
    df_balanced[name] = [x[i] for x in token_features]
for i, name in enumerate(['abs_len_diff','mean_len','longest_substr_ratio']):
    df_balanced[name] = [x[i] for x in length_features]
for i, name in enumerate(['fuzz_ratio','fuzz_partial_ratio','token_sort_ratio','token_set_ratio']):
    df_balanced[name] = [x[i] for x in fuzzy_features]

# --- Model Training ---

# Final list of features used for model input
features = [
    'q1_len', 'q2_len', 'q1_num_words', 'q2_num_words', 'word_common', 'word_total', 'word_share',
    'cwc_min', 'cwc_max', 'csc_min', 'csc_max', 'ctc_min', 'ctc_max',
    'last_word_eq', 'first_word_eq', 'abs_len_diff', 'mean_len', 'longest_substr_ratio',
    'fuzz_ratio', 'fuzz_partial_ratio', 'token_sort_ratio', 'token_set_ratio'
]
X = df_balanced[features]  # Feature matrix
y = df_balanced['is_duplicate']  # Target label

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train the model
model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# --- Save the Model and Features for Later Use ---
joblib.dump(model, "C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/duplicate_analyzer_model.pkl")
joblib.dump(features, "C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/duplicate_analyzer_features_list.pkl")