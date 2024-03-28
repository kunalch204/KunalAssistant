import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

# Function to generate bot response
def generate_response(user_input):
    chain = get_qa_chain()
    response = chain(user_input)
    return response["result"]

st.title("Kunal Chopra Personal Assistant")

# btn = st.button("Create Knowledgebase")
# if btn:
#     create_vector_db()

# Initialize chat history if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
user_input = st.chat_input("Ask Assistant...")

if user_input:
    # Immediately add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display the user's message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Show a loading message while generating the bot's response
    with st.spinner('Generating response...'):
        bot_response = generate_response(user_input)

    # Add the bot's response to the chat history and display it
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)
