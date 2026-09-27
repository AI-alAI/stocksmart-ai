# StockSmart AI

**AI-powered inventory assistant for Moroccan auto-parts shops — ask in English or French, get instant answers.**

Built for **Come Build with AI — GOMYCODE Hackathon, 27 September 2026 (Morocco)**

---

## Problem

Small auto-parts shop owners in Morocco lose sales because they can't answer questions like *"How many oil filters do I have left?"* quickly. They don't have time for complex ERP software.

## Solution

StockSmart AI converts natural language (English, French) into safe, read-only SQLite queries via the Groq API, executes them, and returns clear conversational answers — with the SQL shown for transparency.

## Features

- 🗣️ Natural language queries (English + French, typing or voice)
- 🔒 Strict SQL sanitizer — AI can never modify the database
- 📊 Live inventory dashboard with low/out-of-stock filters
- ➕ Full CRUD from the browser (add, edit, delete products)
- 📥 CSV export
- 📈 Analytics: pie, bar, and interactive 3D charts
- 🎤 Voice input (Groq Whisper) · 🔊 Voice output (browser TTS)
- 🕘 Conversation history

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.13 |
| UI | Streamlit |
| AI / LLM | Groq API (`openai/gpt-oss-120b`) |
| Speech-to-text | Groq Whisper (`whisper-large-v3`) |
| Text-to-speech | Browser Web Speech API |
| Database | SQLite (read-only for AI) |
| Charts | Plotly + Three.js |

## Run Locally

```bash
git clone https://github.com/AI-alAI/stocksmart-ai.git
cd stocksmart-ai
pip install -r requirements.txt

# Add your Groq API key (free at https://console.groq.com/keys)
mkdir .streamlit
echo 'GROQ_API_KEY = "gsk_YOUR_KEY_HERE"' > .streamlit/secrets.toml

# Seed sample inventory
python seed_data.py

# Run
streamlit run app.py
