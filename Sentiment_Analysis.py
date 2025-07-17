# --- Import Required Libraries ---
import re
import nltk
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# --- Ensure Required NLTK Data is Downloaded ---
def safe_nltk_download(package):
    try:
        nltk.data.find(f'corpora/{package}')
    except LookupError:
        nltk.download(package)

safe_nltk_download('stopwords')
safe_nltk_download('wordnet')

# --- Load Pretrained Tokenizer and Model ---
tokenizer = joblib.load("C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/Sentiment_tokenizer.pkl")  # Loading the tokenizer
model = load_model("C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/Sentiment_analyzer_model.h5")  # Loading the model

# --- Preprocessing Function ---
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))

def preprocess(text):  # Fucntion to Clean and normalize raw input text
    text = str(text).lower().strip()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    tokens = text.split()  # Tokenizing the words
    words = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens if word not in STOP_WORDS]  # Lemmatizing the review

    return ' '.join(words)

# --- Prediction Function ---
def predict_sentiment(text):
    cleaned = preprocess(text)  # Clean the input text
    sequence = tokenizer.texts_to_sequences([cleaned])  # Tokenize text
    padded = pad_sequences(sequence, maxlen=300, padding='post')  # Pad sequence
    prediction = model.predict(padded)[0][0]  # Get prediction score (0-1)
    sentiment = "Positive 😀" if prediction > 0.5 else "Negative 😞"
    confidence = round(prediction * 100, 2) if prediction > 0.5 else round((1 - prediction) * 100, 2)
    
    print(f"\nInput: {text}")
    print(f"Predicted Sentiment: {sentiment}")
    print(f"Confidence: {confidence}%")

# --- Example Usage ---
sample_review = input("Enter your review : ")
predict_sentiment(sample_review)
