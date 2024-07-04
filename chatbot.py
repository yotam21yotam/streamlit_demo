import streamlit as st
from transformers import pipeline

# Load a text-generation model
generator = pipeline('text-generation', model='distilgpt2')

st.title('About Me Chatbot')

st.write('I am a chatbot trained to answer questions about Yotam! Ask me anything.')

user_input = st.text_input("Type your question here:")

if user_input:
    # Generate response using the model
    responses = generator(user_input, max_length=50, num_return_sequences=1)
    response = responses[0]['generated_text']
    st.text_area("Response", value=response, height=150, max_chars=None, help="Response from the chatbot.")
