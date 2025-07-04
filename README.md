# Intelligent AI Chatbot

-   **Developed by:** Ayush Patil
-   **Intern ID:** CT06DN877
-   **Domain:** Python Development
-   **Duration:** 6 weeks
-   **Mentor:** Neela Santosh
-   **Company:** Code Tech IT Solutions

This project showcases an intelligent AI chatbot with a sophisticated user interface and a robust Python backend for natural language processing and response generation.

## Features

-   **Interactive Chat Interface:** A sleek, modern web interface built with HTML and Tailwind CSS, featuring glassmorphism effects, animated backgrounds, and responsive design.
-   **Animated Background:** Dynamic particles and a subtle grid pattern create an engaging visual experience.
-   **Gradient Effects:** Utilizes CSS gradients for titles, buttons, and message bubbles, enhancing visual appeal.
-   **Glassmorphism UI:** Implements trendy glassmorphism effects for various UI elements, providing a frosted, translucent look.
-   **Typing Indicator:** Displays a "AI is thinking..." animation when the bot is generating a response.
-   **Quick Action Buttons:** Pre-defined buttons for common queries, improving user experience.
-   **Advanced AI Backend (Python/Flask):**
    -   **NLTK Integration:** Utilizes NLTK for text preprocessing, tokenization, lemmatization, and stop word removal.
    -   **TF-IDF Vectorization:** Employs TF-IDF for converting text into numerical representations.
    -   **Cosine Similarity:** Calculates the similarity between user input and the knowledge base to find the best response.
    -   **Intent Classification:** Classifies user intent (e.g., greetings, questions about Python, AI, technology) to provide more accurate and contextually relevant responses.
    -   **Extensive Knowledge Base:** Contains a rich set of pre-defined responses for various topics like programming, AI, technology, science, history, geography, health, food, movies, sports, and music.
    -   **Dynamic Fallback Responses:** Provides helpful fallback messages when the AI doesn't have a direct answer.
    -   **Flask API:** Serves the chatbot's responses via a simple REST API.

## Website Demo

Check out a quick demonstration of the chatbot's interactive interface and functionalities below:

![Chatbot in Action](path/to/your/screen_recording.gif)

*Note: The GIF above is a screen recording of the chatbot's web interface, showcasing its dynamic responses and user interaction.*

## Technologies Used

### Frontend

-   **HTML5:** Structure of the web application.
-   **Tailwind CSS:** A utility-first CSS framework for rapid UI development and styling.
-   **JavaScript:** For interactive elements, DOM manipulation, sending messages, handling responses, and animations.

### Backend

-   **Python:** The core programming language for the AI logic.
-   **Flask:** A micro web framework for building the REST API.
-   **NLTK (Natural Language Toolkit):** For natural language processing tasks.
-   **Scikit-learn:** For TF-IDF vectorization and cosine similarity calculations.
-   **NumPy:** For numerical operations.

## How It Works

1.  **Frontend (index.html):**
    * The user interacts with the chat interface by typing messages or clicking quick action buttons.
    * JavaScript captures the user's input and sends it to the Flask backend via an asynchronous (AJAX) request.
    * It also manages the display of messages, including user messages, bot responses, and a typing indicator.
    * Animated particles and a grid background are dynamically generated and styled using CSS and JavaScript for an immersive experience.

2.  **Backend (main.py):**
    * **Initialization:** The `IntelligentChatbot` class initializes NLTK components (lemmatizer, stopwords) and builds a comprehensive knowledge base from predefined categories and additional question-answer pairs. It also fits a `TfidfVectorizer` to this training data.
    * **Preprocessing:** User input is cleaned by converting it to lowercase, removing punctuation, tokenizing, removing stopwords, and lemmatizing the words.
    * **Intent Classification:** The chatbot first attempts to classify the user's intent based on a set of keywords (e.g., "hello" for greetings, "python" for programming). If a clear intent is found, a random response from that category is returned.
    * **Similarity Search (TF-IDF & Cosine Similarity):** If no direct intent is matched, the preprocessed user input is transformed into a TF-IDF vector. This vector is then compared against the TF-IDF matrix of the entire knowledge base using cosine similarity to find the most semantically similar question.
    * **Response Generation:** The response corresponding to the best-matched question in the knowledge base is returned. A similarity threshold is applied to ensure the relevance of the response; otherwise, a general fallback message is provided.
    * **Flask API:** The Flask application exposes a `/chat` endpoint that receives user messages and returns the chatbot's generated response. It also serves the `index.html` file.

## Setup and Installation

### Prerequisites

-   Python 3.x
-   `pip` (Python package installer)

### Installation Steps

1.  **Clone the repository (or download the files):**
    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    * On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required Python packages:**
    ```bash
    pip install nltk scikit-learn Flask Flask-Cors numpy
    ```

5.  **Download NLTK data:**
    The `main.py` script includes checks to download necessary NLTK data (`punkt`, `stopwords`, `wordnet`) if they are not already present.

## Running the Chatbot

1.  **Start the Flask backend:**
    ```bash
    python main.py
    ```
    You should see output indicating that the chatbot is starting and the web server is running, typically on `http://localhost:5000`.

2.  **Open the web interface:**
    Navigate to `http://localhost:5000` in your web browser.

## Customization

-   **Knowledge Base:**
    * Modify the `self.knowledge_base` dictionary in `main.py` to add, remove, or update topics and their corresponding responses.
    * Add more `(question, answer)` pairs to the `additional_training` list for a broader range of specific responses.
-   **UI Styling:**
    * Adjust the `style` block in `index.html` or extend it with more Tailwind CSS classes to change the visual appearance.
    * Modify the CSS variables in `:root` to easily change color schemes and gradients.
-   **Bot Responses:**
    * Refine the `getBotResponse` function in `index.html` (for quick client-side responses) and the `get_response` function in `main.py` (for AI responses) to enhance the chatbot's personality and accuracy.
-   **Particle Effects:**
    * Modify the `createParticles` function and associated CSS in `index.html` to change the number, size, color, or animation of the background particles.

## Credits

-   **Developed by:** Ayush Patil
-   **Mentor:** Neela Santosh
-   **Company:** Code Tech IT Solutions
