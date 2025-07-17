# --- Core Libraries for Data Processing ---
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import re  # For text cleaning
import nltk  # Natural Language Toolkit
import warnings  # To suppress warning messages
import joblib  # For saving/loading Python objects

# --- NLP-Specific Libraries ---
from bs4 import BeautifulSoup  # Used to remove HTML tags
from nltk.corpus import stopwords  # List of English stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer  # Lemmatization and stemming

# --- Deep Learning Libraries (Keras/TensorFlow) ---
from tensorflow.keras.models import Model  # Keras Functional API
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Dense, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer  # For text tokenization
from tensorflow.keras.preprocessing.sequence import pad_sequences  # For padding sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # For stopping early and saving best model

# --- Evaluation and Training Utilities ---
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.metrics import classification_report, accuracy_score  # For model evaluation

# --- Download required NLTK packages (stopwords, wordnet) ---
def safe_nltk_download(package):
    try:
        nltk.data.find(f'corpora/{package}')
    except LookupError:
        nltk.download(package)

safe_nltk_download('stopwords')  # Ensure stopwords are available
safe_nltk_download('wordnet')  # Ensure WordNet is available

# --- Ignore warnings to keep logs clean ---
warnings.filterwarnings('ignore')

# --- Load IMDb Dataset ---
df = pd.read_csv("C:/Users/ankit/OneDrive/Documents/Personal/Repls/Raw Code/Sentiment analyzer/Dataset/IMDB Dataset.csv")
df.columns = ['text', 'sentiment']  # Rename columns for clarity
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})  # Convert sentiment to binary labels

# --- Text Preprocessing ---
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))  # Load stopwords into a set

def preprocess(text):
    """
    Cleans the input review:
    - Lowercases
    - Removes HTML and non-alphabetic characters
    - Tokenizes, removes stopwords
    - Applies lemmatization and stemming
    """
    text = str(text).lower().strip()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-letter characters
    tokens = text.split()
    words = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens if word not in STOP_WORDS]
    return ' '.join(words)

# Apply preprocessing to all reviews
df['clean_text'] = df['text'].apply(preprocess)

# --- Tokenization ---
VOCAB_SIZE = 20000  # Limit vocabulary size for memory/speed balance
MAX_LEN = 300  # Maximum review length to pad/truncate

tokenizer = Tokenizer(num_words=VOCAB_SIZE)  # Initialize tokenizer
tokenizer.fit_on_texts(df['clean_text'])  # Learn word index from training text
sequences = tokenizer.texts_to_sequences(df['clean_text'])  # Convert text to integer sequences
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')  # Pad sequences with 0s
word_index = tokenizer.word_index  # Get word:index mapping

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, df['label'].values, test_size=0.2, random_state=42)  # Split dataset

# --- Build Model: Trainable Embeddings (No GloVe) + Conv1D ---
input_seq = Input(shape=(MAX_LEN,))  # Input layer for sequences

# Trainable embedding layer (learns embeddings during training)
embedding_layer = Embedding(input_dim=VOCAB_SIZE + 1,
                             output_dim=100,
                             input_length=MAX_LEN,
                             trainable=True)(input_seq)

# Apply 1D convolution for feature extraction
x = Conv1D(128, kernel_size=5, activation='relu')(embedding_layer)  # Feature extractor
x = BatchNormalization()(x)  # Normalize layer outputs
x = GlobalMaxPooling1D()(x)  # Convert to fixed-length vector
x = Dropout(0.5)(x)  # Dropout for regularization
x = Dense(64, activation='relu')(x)  # Hidden dense layer
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
out = Dense(1, activation='sigmoid')(x)  # Output layer for binary classification

# Compile the model
model = Model(inputs=input_seq, outputs=out)  # Create model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # Compile with optimizer & loss
model.summary()  # Display model architecture

# --- Callbacks: EarlyStopping + Save Best Model ---
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)  # Stop if no improvement
checkpoint = ModelCheckpoint("best_model_trained_embedding.h5", monitor='val_accuracy', save_best_only=True)  # Save best model

# --- Train the Model ---
model.fit(X_train, y_train,
          batch_size=64,
          epochs=10,
          validation_split=0.1,
          callbacks=[early_stop, checkpoint])

# --- Evaluate on Test Set ---
y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Convert probabilities to binary predictions
print("Accuracy Score:", accuracy_score(y_test, y_pred))  # Accuracy
print("Classification Report:\n", classification_report(y_test, y_pred))  # Detailed classification report

# --- Save Final Model and Tokenizer ---
model.save("C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/Sentiment_analyzer_model.h5")  # Save final model
joblib.dump(tokenizer, "C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/Sentiment_tokenizer.pkl")  # Save tokenizer
