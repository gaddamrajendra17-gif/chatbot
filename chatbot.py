"""Core chatbot logic using Ollama + Llama 3.2.

- Imported by app.py (Streamlit web UI)
- Can also be run directly as a terminal chatbot:  python chatbot.py

Terminal commands:
  /clear  - reset conversation memory
  /exit   - quit (Ctrl+C also works)
"""

import os
import sys
from typing import Iterator

from dotenv import load_dotenv
from ollama import Client, ResponseError

load_dotenv()

HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))


class ChatBot:
    """Keeps conversation history and streams replies from an Ollama model."""

    def __init__(
        self,
        model: str = MODEL,
        host: str = HOST,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = TEMPERATURE,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.client = Client(host=host)
        self.history = []
        self.clear()

    def clear(self) -> None:
        """Reset the conversation to just the system prompt."""
        self.history = [{"role": "system", "content": self.system_prompt}]

    def load_history(self, messages: list) -> None:
        """Restore a saved conversation (list of {'role', 'content'} dicts)."""
        self.clear()
        self.history.extend(messages)

    def stream(self, user_message: str) -> Iterator[str]:
        """Add a user message, yield the reply piece by piece, then save it.

        If the model call fails, the user message is removed from history
        and the exception is re-raised so the caller can show an error.
        """
        self.history.append({"role": "user", "content": user_message})
        reply = ""
        try:
            response = self.client.chat(
                model=self.model,
                messages=self.history,
                stream=True,
                options={"temperature": self.temperature},
            )
            for chunk in response:
                piece = chunk["message"]["content"]
                reply += piece
                yield piece
        except Exception:
            self.history.pop()
            raise
        self.history.append({"role": "assistant", "content": reply})

    def ask(self, user_message: str) -> str:
        """Non-streaming helper: return the full reply as one string."""
        return "".join(self.stream(user_message))

    def list_models(self) -> list:
        """Names of models installed in the local Ollama server."""
        try:
            data = self.client.list()
            models = data["models"] if isinstance(data, dict) else data.models
            names = []
            for m in models:
                name = m["model"] if isinstance(m, dict) else m.model
                names.append(name)
            return names
        except Exception:
            return []


def main() -> int:
    bot = ChatBot()
    print(f"Chatbot ready (model: {bot.model}). Type /exit to quit, /clear to reset.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            return 0

        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit"):
            print("Goodbye!")
            return 0
        if user_input.lower() == "/clear":
            bot.clear()
            print("Conversation cleared.\n")
            continue

        print("Bot: ", end="", flush=True)
        try:
            for piece in bot.stream(user_input):
                print(piece, end="", flush=True)
            print("\n")
        except ResponseError as e:
            print(f"\n[Model error] {e.error}")
            print(f"Try: ollama pull {bot.model}\n")
        except Exception as e:
            print(f"\n[Connection error] {e}")
            print("Is Ollama running? Start it with: ollama serve\n")


if __name__ == "__main__":
    sys.exit(main())