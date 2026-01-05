import streamlit as st
import json
import requests
import os
from dotenv import load_dotenv

# ---------------- Load Env ----------------
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# ---------------- Page Config ----------------
st.set_page_config(page_title="Cyber Risk AI Chatbot", layout="wide")
st.title("🛡️ Cyber Risk Assessment AI Chatbot")

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    with open("data.json", "r") as f:
        return json.load(f)

data = load_data()

# ---------------- Chat History ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- User Input ----------------
user_input = st.chat_input("Ask anything about vulnerabilities...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Context for AI
    context = json.dumps(data, indent=2)

    prompt = f"""
You are a cybersecurity risk analysis assistant.

Using the following vulnerability data:
{context}

Answer the user's question clearly and professionally.

User question:
{user_input}
"""

    # Call OpenRouter API
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful cybersecurity expert."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"]["content"]
    else:
        answer = "⚠️ Error communicating with AI service."

    # Show assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
