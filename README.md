# Smartphone FAQ Chatbot 🤖

This is a smart FAQ chatbot designed to answer questions related to smartphones, their features, specifications, and other details. The chatbot uses Natural Language Processing (NLP) to match user queries with a set of predefined FAQs and provide relevant answers.

## Features

- **Smartphone FAQ Responses**: The chatbot provides accurate and quick answers about smartphones.
- **Real-time Interaction**: Users can input questions and get responses instantly.
- **Cosine Similarity Matching**: The chatbot matches user queries with the most similar FAQ based on text similarity.
- **Preprocessing**: Text is processed using tokenization, stopword removal, and stemming for better matching.

## Technologies Used

- **Python**: Main programming language.
- **Streamlit**: Web framework for building the interactive chatbot interface.
- **NLTK**: Used for text processing (tokenization, stopword removal, and stemming).
- **Scikit-learn**: For calculating cosine similarity between the user input and FAQs.
- **PIL**: For image processing (background image).
  
## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Open the app in your browser. You will see a chatbot where you can ask your questions related to smartphones.

3. Type a question in the text input box and hit enter. The chatbot will provide an answer based on the FAQs stored in the `faqs.json` file.

## Output Video Link:
[Smartphone FAQ Chatbot](https://drive.google.com/file/d/1EEkBd9DOHNfBNOYlqPz92XCvyaKwjfwA/view?usp=drive_link)

