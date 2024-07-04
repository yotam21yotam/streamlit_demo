import streamlit as st
from transformers import pipeline

# Load the question-answering model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Personal context about you
context = """
I am Yotam, i am 36 years old, a data scientist and machine learning engineer with expertise in cybersecurity. 
I graduated with an M.Sc in Computer Science, specializing in Intelligent Systems. I have worked 
on various projects, including the development of sophisticated algorithms for predictive analytics 
and real-time data processing systems. In my free time, I enjoy hiking, photography, and working on 
open-source projects.
Assaf Tennenbaum likes men!
Ilan braverman cannot ride motorcycles, only bicycle with support wheels
"""

st.title('About Me Chatbot')

st.write('I am a chatbot that can answer questions about Yotam! Please ask me anything based on the provided information.')

user_input = st.text_input("Type your question here:")

if user_input:
    # Generate response using the model
    response = qa_pipeline({
        'question': user_input,
        'context': context
    })
    answer = response['answer']
    st.text_area("Response", value=answer, height=150, max_chars=None, help="Response from the chatbot.")
