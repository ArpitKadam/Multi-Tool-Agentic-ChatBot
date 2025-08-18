<p align="center">
  <a href="https://github.com/ArpitKadam/Multi-Tool-Agentic-ChatBot">
    <img src="https://www.goseeko.com/blog/wp-content/uploads/2025/07/image-2.png" alt="Multi-Tool Agentic ChatBot" width="650"/>
  </a>
</p>

<h1 align="center" style="font-size:2.8em; font-weight:500;">
  🤖 Multi-Tool Agentic ChatBot 🛠️
</h1>

<p align="center">
  <i>
    A fully agentic, multi-tool, state-based conversational AI application.<br>
    Modular, extensible, and built for rapid tool augmentation.
  </i>
</p>

---

**Demo** :- https://multi-tool-agentic-chatbot.onrender.com/

---

## 🚀 Features

- **Agentic Chatbot Core:** Interactive chatbot harnessing LLMs and agentic workflows.
- **Multi-Tool Support:** Integrated tools for tasks like AI news summarization, Wikipedia lookup, PDF/Arxiv parsing, search, and more.
- **Plug & Play Tools:** Easily add new tools (code, search, summarization, etc.) via declarative architecture.
- **Modern Stack:** Utilizes LangChain, LangGraph, and a strong Python foundation.
- **Web UI & API:** Easily expose the chatbot for web or API usage.

---

## 🗂️ Project Structure
```
Multi-Tool-Agentic-ChatBot/
│
├── app.py # Entrypoint to launch the chatbot app (CLI/server)
├── requirements.txt # Project dependencies
├── LICENSE
├── README.md
├── src/
│ └── langgraph/
│ ├── init.py
│ ├── main.py # Core app orchestration
│ ├── graph/ # Graph logic and flow definition
│ ├── nodes/ # Custom chat nodes (eg. tool routers, selectors, etc)
│ ├── state/ # State management for chat, tool use, etc
│ ├── llms/ # LLM connectors/wrappers
│ ├── tools/ # Individual tool integrations
│ └── ui/ # Web or streamlit UI logic (optional/expandable)
│
├── .vscode/ # VS Code workspace config
```

---

## 🛠️ Installation

### Requirements

- Python >= 3.10
- pip
- API keys (.env) for any enabled LLMs/tools (OpenAI, Groq, Tavily, SerpAPI, etc)

### 1. Clone the repository
```
git clone https://github.com/ArpitKadam/Multi-Tool-Agentic-ChatBot.git
cd Multi-Tool-Agentic-ChatBot
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Set up environment

Place your API keys in a `.env` file at the project root as required by your tools/LLMs.

---

## 🧑‍💻 Usage

### CLI
```
python app.py
```

_By default, this runs the chat app main cycle using the agentic orchestration in `src/langgraph/main.py`._

### Web UI

If the `ui` module uses Streamlit (recommended for local demo):
```
streamlit run app.py
```

_For a full-featured API, adapt `app.py` to spin up FastAPI or similar if required._

---

## 🔌 Extending & Customizing

- **Add New Tools:** Place tool code in `src/langgraph/tools/`, wire it into the node/graph logic.
- **Custom Agents:** Edit or build new nodes in `src/langgraph/nodes/`.
- **Workflow:** Modify `src/langgraph/graph/` or `main.py` for custom chat/agentic flows.
- **UI/Backend:** Connect to Streamlit, FastAPI, or any relevant frontend in `src/langgraph/ui/`.

---

## 🧰 Example Tools (from requirements.txt)

- Wikipedia Search
- DuckDuckGo or Google Search
- AI News Summarizer
- PDF/Arxiv parsing
- Faiss/Chroma vector search
- OpenAI, Groq, Ollama, Cohere, Nvidia endpoints
- Streamlit, Rich console, etc

_(See requirements.txt and the `tools/` folder for the full list)_

---

## 📄 License

This project is licensed under the Apache 2.0 License. See [LICENSE](https://github.com/ArpitKadam/Multi-Tool-Agentic-ChatBot/blob/main/README.md).

---

## 👤 Author

- Arpit Kadam ([GitHub](https://github.com/ArpitKadam))

---

> _Easily evolve your LLM chatbot with new tools and agentic reasoning… all in Python!_
