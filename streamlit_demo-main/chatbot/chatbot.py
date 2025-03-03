import streamlit as st
from transformers import pipeline

# Use only one model to save memory
model_path = "distilbert-base-uncased-distilled-squad"

# Personal context about Yotam
context = """
Kobi Edri is team leader in elal from Mevasere Zion

Elad Dgani is the CDO and he like beer 




Kobi Edri is team leader in elal from Mevasere Zion

Elad Dgani is the CDO and he like beer 
"""

st.title('Yotam\'s Chatbot')

st.write('I am a chatbot that can answer questions about Yotam! Please ask me anything you would like to know about him:')

# Load the question-answering model
qa_pipeline = pipeline("question-answering", model=model_path)

user_input = st.text_input("Type your question here:")

if user_input:
    try:
        # Generate response using the model
        response = qa_pipeline({'question': user_input, 'context': context})
        answer = response['answer']
        st.text_area("Response", value=answer, height=150, max_chars=None, help="Response from the chatbot.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
