import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# ----------------- LOAD ENV -----------------
load_dotenv()

# ----------------- LLM -----------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ----------------- PROMPT -----------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Remember previous conversations."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# ----------------- CHAIN -----------------
chain = prompt | llm

# ----------------- MEMORY STORE -----------------
store = {}

if "store" not in st.session_state:
    st.session_state["store"] = store

store = st.session_state["store"]

# ----------------- SESSION HISTORY -----------------
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# ----------------- STREAMLIT UI -----------------
st.title("Chatbot with Memory (LangChain + Groq)")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = "default"

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ----------------- DISPLAY CHAT HISTORY -----------------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------- USER INPUT -----------------
user_input = st.chat_input("Type your message here...")

if user_input:
    # store user message
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # invoke chain with memory
    response = chain_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": st.session_state["session_id"]}}
    )

    bot_reply = response.content

    # store assistant message
    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})

    st.rerun()

# ----------------- RESET BUTTON -----------------
if st.button("Reset Conversation"):
    st.session_state["messages"] = []
    st.session_state["store"] = {}
    store.clear()