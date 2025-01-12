import json
import streamlit as st
import nltk    
from PIL import Image
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data (make sure these resources are available)

nltk.download('punkt_tab')  # Ensure you specify the correct directory if needed
nltk.download('stopwords')

# Fallback stopwords in case NLTK download fails
stop_words_fallback = set([  
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
    'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
    'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 
    'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
    'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 
    'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
    's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 
    'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 
    'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 
    'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 
    'wouldn'
])

# Load FAQs from external JSON file
def load_faqs(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Preprocess text using NLTK
def preprocess_text(text):
    # Initialize tokens to an empty list to prevent the UnboundLocalError
    tokens = []

    try:
        # Tokenize the text
        tokens = word_tokenize(text.lower())

        # Attempt to use NLTK stopwords
        stop_words = set(stopwords.words('english'))
    except Exception as e:
        # If NLTK stopwords fail or tokenization fails, use fallback
        print(f"Error in text processing: {e}")
        stop_words = stop_words_fallback
    
    # Remove stopwords and non-alphanumeric tokens
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # Stem the words
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    return " ".join(stemmed_tokens)

# Load the FAQs
faqs = load_faqs("data/faqs.json")

# Preprocess FAQs
questions = [preprocess_text(q) for q in faqs.keys()]
answers = list(faqs.values())

# Streamlit app
st.set_page_config(page_title="Smartphone FAQ Chatbot 🤖", page_icon=":iphone:")
st.markdown("<h1 style='text-align: center; color: white;'>Smartphone FAQ Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: white;'>Ask your questions about smartphone features, specifications, and details!</h3>", unsafe_allow_html=True)

# Add background image related to smartphones (using st.image)
background_image_path = "C:/Users/ADMIN/Downloads/faq_image.png"  # Replace with your image URL or local file path
image = Image.open(background_image_path)
image = image.resize((500, 400), Image.Resampling.LANCZOS)

# Display the resized image
st.image(image, use_container_width=True)

# Apply the CSS class to the container where the image is placed
st.markdown('<div class="background-image"></div>', unsafe_allow_html=True)


# User input section below the image
st.markdown("<div style='text-align: center;'><h4 style='color: white;'>Your Question:</h4></div>", unsafe_allow_html=True)
user_query = st.text_input("Enter your question", key="user_input", label_visibility="collapsed")

if user_query:
    # Preprocess user query
    processed_query = preprocess_text(user_query)

    # NLP processing
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions)
    user_vec = vectorizer.transform([processed_query])
    
    # Find closest match
    similarities = cosine_similarity(user_vec, vectors).flatten()
    max_idx = similarities.argmax()
    
    if similarities[max_idx] > 0.5:  # Confidence threshold
        response = answers[max_idx]
    else:
        response = "Sorry, I couldn't find an answer to your question. Please try asking in a different way."
    
    # Display response
    st.write("**Bot:**", response)
