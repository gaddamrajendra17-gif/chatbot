"""Streamlit web UI for the Ollama + Llama 3.2 chatbot.

Run with:  streamlit run app.py

"""

import streamlit as st
from ollama import ResponseError

from chatbot import MODEL, SYSTEM_PROMPT, TEMPERATURE, ChatBot

st.set_page_config(page_title="Llama 3.2 Chatbot", page_icon="🦙", layout="centered")

# ---------- Sidebar: settings ----------
with st.sidebar:
    st.header("Settings")

    # Discover installed models; fall back to the one in .env
    probe = ChatBot()
    installed = probe.list_models() or [MODEL]
    default_index = installed.index(MODEL) if MODEL in installed else 0

    model = st.selectbox("Model", installed, index=default_index)
    temperature = st.slider("Temperature", 0.0, 1.0, TEMPERATURE, 0.05)
    system_prompt = st.text_area("System prompt", SYSTEM_PROMPT, height=120)

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Make sure Ollama is running: `ollama serve`")

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content"} (no system msg)

# ---------- Header ----------
st.title("🦙 Llama 3.2 Chatbot")
st.caption(f"Running locally via Ollama · model: `{model}`")

# ---------- Show past messages ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Handle new input ----------
if prompt := st.chat_input("Type your message..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    # Rebuild the bot each run with current sidebar settings + saved history
    bot = ChatBot(model=model, system_prompt=system_prompt, temperature=temperature)
    bot.load_history(st.session_state.messages)

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(bot.stream(prompt))
        except ResponseError as e:
            reply = None
            st.error(f"Model error: {e.error}\n\nTry: `ollama pull {model}`")
        except Exception as e:
            reply = None
            st.error(f"Could not reach Ollama: {e}\n\nStart it with: `ollama serve`")

    if reply is not None:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": reply})