import nltk
import string
import random
import json
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

class IntelligentChatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Knowledge base with various topics
        self.knowledge_base = {
            "greetings": [
                "Hello! How can I help you today?",
                "Hi there! What would you like to know?",
                "Hey! I'm here to assist you.",
                "Greetings! How may I be of service?"
            ],
            "farewells": [
                "Goodbye! Have a great day!",
                "See you later! Take care!",
                "Bye! Feel free to chat anytime!",
                "Farewell! It was nice talking with you!"
            ],
            "thanks": [
                "You're very welcome!",
                "Happy to help!",
                "No problem at all!",
                "Glad I could assist you!"
            ],
            "python": [
                "Python is a high-level, interpreted programming language known for its simplicity and readability.",
                "Python was created by Guido van Rossum and first released in 1991.",
                "Python is widely used for web development, data science, artificial intelligence, and automation.",
                "Python's philosophy emphasizes code readability and simplicity, following the principle 'There should be one obvious way to do it.'"
            ],
            "ai": [
                "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.",
                "AI includes machine learning, deep learning, natural language processing, and computer vision.",
                "AI is transforming industries like healthcare, finance, transportation, and entertainment.",
                "The goal of AI is to create systems that can perform tasks that typically require human intelligence."
            ],
            "technology": [
                "Technology refers to the application of scientific knowledge for practical purposes.",
                "Modern technology includes computers, smartphones, internet, and various digital tools.",
                "Technology has revolutionized communication, education, healthcare, and business.",
                "Emerging technologies include quantum computing, blockchain, IoT, and biotechnology."
            ],
            "weather": [
                "I don't have access to real-time weather data, but you can check weather apps or websites for current conditions.",
                "Weather patterns are influenced by factors like temperature, humidity, pressure, and wind.",
                "Climate change is affecting global weather patterns and causing more extreme weather events.",
                "Meteorology is the science that studies atmospheric conditions to predict weather."
            ],
            "science": [
                "Science is a systematic method of understanding the natural world through observation and experimentation.",
                "Major branches of science include physics, chemistry, biology, and earth sciences.",
                "The scientific method involves hypothesis formation, testing, and peer review.",
                "Science has led to countless innovations that improve human life and understanding."
            ],
            "history": [
                "The Roman Empire was founded in 27 BC and lasted for over 1000 years.",
                "World War II began in 1939 with the German invasion of Poland.",
                "The Declaration of Independence was signed on July 4, 1776, in the United States.",
                "Ancient Egypt is known for its pharaohs, pyramids, and hieroglyphs."
            ],
            "geography": [
                "Mount Everest is the highest mountain in the world, located in the Himalayas.",
                "The Amazon River in South America is the largest river by discharge volume.",
                "The Sahara Desert is the largest hot desert in the world, spanning much of North Africa.",
                "Australia is both a continent and a country."
            ],
            "health": [
                "Regular exercise and a balanced diet are key to maintaining good health.",
                "Getting enough sleep, typically 7-9 hours for adults, is crucial for well-being.",
                "Hydration is important; aim to drink plenty of water throughout the day.",
                "Stress management techniques like meditation or yoga can improve overall health."
            ],
            "food": [
                "Pizza originated in Italy and is now a popular dish worldwide.",
                "Sushi is a traditional Japanese dish of prepared vinegared rice.",
                "Curry is a dish with a sauce seasoned with spices, common in South Asian cuisine.",
                "Chocolate comes from cacao beans and is a widely loved sweet treat."
            ],
            "movies": [
                "The Shawshank Redemption is often cited as one of the greatest films ever made.",
                "Science fiction, fantasy, drama, and comedy are popular movie genres.",
                "Filmmaking involves screenwriting, directing, acting, and post-production.",
                "Streaming services have revolutionized how people watch movies and TV shows."
            ],
            "sports": [
                "Football (soccer) is the most popular sport globally.",
                "The Olympic Games are a major international multi-sport event.",
                "Basketball, tennis, and swimming are other widely recognized sports.",
                "Regular physical activity through sports offers numerous health benefits."
            ],
            "books": [
                "To Kill a Mockingbird by Harper Lee is a classic of modern American literature.",
                "Books can transport readers to different worlds and perspectives.",
                "Reading improves vocabulary, critical thinking, and empathy.",
                "There are many genres of books, including fiction, non-fiction, poetry, and fantasy."
            ],
            "music": [
                "Music is a universal language with countless genres and styles.",
                "Different instruments, from guitars to pianos, create diverse sounds.",
                "Music can evoke emotions, tell stories, and be used for celebration or relaxation.",
                "Many cultures have unique musical traditions and instruments."
            ]
        }
        
        # Create comprehensive training data
        self.training_data = []
        self.responses = []
        
        # Add knowledge base to training data
        for category, responses in self.knowledge_base.items():
            for response in responses:
                self.training_data.append(category)
                self.responses.append(response)
        
        # Add more training examples
        additional_training = [
            ("what is python", "Python is a versatile programming language that's perfect for beginners and experts alike. It's used in web development, data science, AI, and more!"),
            ("how are you", "I'm doing great, thank you for asking! I'm here and ready to help you with any questions you have."),
            ("tell me about yourself", "I'm an AI chatbot created to help answer your questions and have conversations. I use natural language processing to understand and respond to you!"),
            ("what can you do", "I can answer questions about various topics like technology, science, programming, and general knowledge. I can also have casual conversations with you!"),
            ("help", "I'm here to help! You can ask me questions about technology, science, programming, or just chat with me. What would you like to know?"),
            ("programming languages", "There are many programming languages like Python, JavaScript, Java, C++, and more. Each has its own strengths and use cases."),
            ("machine learning", "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed."),
            ("data science", "Data science combines statistics, programming, and domain expertise to extract insights from data and make data-driven decisions."),
            ("when was rome founded", "The Roman Empire was founded in 27 BC."),
            ("who won world war 2", "World War II was fought by the Allied Powers (including the United States, United Kingdom, and Soviet Union) against the Axis Powers (Germany, Italy, and Japan), with the Allies ultimately winning."),
            ("tallest mountain", "Mount Everest is the tallest mountain in the world."),
            ("largest river", "The Amazon River is the largest river by discharge volume."),
            ("how to stay healthy", "To stay healthy, focus on regular exercise, a balanced diet, adequate sleep, and stress management."),
            ("what is pizza", "Pizza is an Italian dish consisting of a usually round, flat base of leavened wheat-based dough topped with tomatoes, cheese, and various other ingredients (anchovies, olives, meat, etc.), which is then baked at a high temperature."),
            ("favorite movie", "As an AI, I don't have preferences, but The Shawshank Redemption is often cited as a highly acclaimed film!"),
            ("what is football", "Football (or soccer) is the world's most popular sport, played between two teams of 11 players with a ball. The game is played on a rectangular field called a pitch, with a goal at each end."),
            ("best book to read", "It depends on your interests! Some widely acclaimed books include 'To Kill a Mockingbird', '1984', 'The Great Gatsby', and 'Pride and Prejudice'."),
            ("tell me about music", "Music is an art form whose medium is sound. It involves elements like rhythm, melody, harmony, and timbre, and it plays a significant role in human culture and expression."),
            ("what is ai used for", "AI is used in various applications like self-driving cars, virtual assistants, medical diagnosis, fraud detection, and personalized recommendations."),
            ("who is guido van rossum", "Guido van Rossum is the creator of the Python programming language."),
            ("what is nlp", "Natural Language Processing (NLP) is a field of AI that enables computers to understand, interpret, and generate human language."),
            ("how does machine learning work", "Machine learning works by training algorithms on data to identify patterns, make predictions, or take decisions without explicit programming."),
            ("blockchain explained", "Blockchain is a distributed, immutable ledger that records transactions in a secure and transparent way."),
            ("what is IoT", "IoT stands for the Internet of Things, referring to the network of physical objects embedded with sensors, software, and other technologies for the purpose of connecting and exchanging data with other devices and systems over the internet."),
            # Add many more question-answer pairs
        ]
        
        for question, response in additional_training:
            self.training_data.append(question)
            self.responses.append(response)
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.training_data)
    
    def preprocess_text(self, text):
        """Clean and preprocess input text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def classify_intent(self, user_input):
        """Classify user intent based on keywords"""
        user_input_lower = user_input.lower()
        
        # Greeting patterns
        if any(word in user_input_lower for word in ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon']):
            return 'greetings'
        
        # Farewell patterns
        if any(word in user_input_lower for word in ['bye', 'goodbye', 'see you', 'farewell', 'exit', 'quit']):
            return 'farewells'
        
        # Thanks patterns
        if any(word in user_input_lower for word in ['thank', 'thanks', 'appreciate']):
            return 'thanks'
        
        # Topic-based classification
        if any(word in user_input_lower for word in ['python', 'programming', 'code', 'script']):
            return 'python'
        
        if any(word in user_input_lower for word in ['ai', 'artificial intelligence', 'machine learning', 'ml']):
            return 'ai'
        
        if any(word in user_input_lower for word in ['technology', 'tech', 'computer', 'digital']):
            return 'technology'
        
        if any(word in user_input_lower for word in ['weather', 'rain', 'sunny', 'temperature']):
            return 'weather'
        
        if any(word in user_input_lower for word in ['science', 'scientific', 'research', 'experiment']):
            return 'science'
        
        return None
    
    def get_response(self, user_input):
        """Generate response based on user input using multiple approaches"""
        if not user_input.strip():
            return "I didn't catch that. Could you please say something?"
        
        # First, try intent classification
        intent = self.classify_intent(user_input)
        if intent and intent in self.knowledge_base:
            return random.choice(self.knowledge_base[intent])
        
        # If no direct intent match, use similarity search
        processed_input = self.preprocess_text(user_input)
        
        # Transform input using the same vectorizer
        input_vector = self.vectorizer.transform([processed_input])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(input_vector, self.tfidf_matrix).flatten()
        
        # Get the best match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        # Set a threshold for similarity
        if best_similarity > 0.1:
            return self.responses[best_match_idx]
        else:
            # Fallback responses for unrecognized input
            fallback_responses = [
                "That's an interesting question! I'd love to learn more about that topic.",
                "I'm not sure about that specific topic, but I'm always learning! Can you tell me more?",
                "That's something I haven't encountered before. Could you rephrase your question?",
                "I'm still learning about that topic. Is there something else I can help you with?",
                "That's a great question! While I don't have specific information about that, I'm here to help with other topics."
            ]
            return random.choice(fallback_responses)

# Initialize the chatbot
chatbot = IntelligentChatbot()

# Flask web application
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    response = chatbot.get_response(user_message)
    return jsonify({'response': response})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'Chatbot is running!'})

if __name__ == '__main__':
    print("🤖 Intelligent Chatbot is starting...")
    print("📚 Loading NLTK resources...")
    print("🌐 Starting web server...")
    print("💬 Chatbot is ready! Visit http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)