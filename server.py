import streamlit as st
from medgpt import med_gpt

# Streamlit interface
st.title("Chatbot")

user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        # Get response from the chatbot model
        response = med_gpt(user_input)
        st.text("Bot: " + response[0]['generated_text'])
    else:
        st.warning("Please enter something.")