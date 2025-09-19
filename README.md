# HAS - Your AI PDF Sidekick 😎

**HAS** is a local AI assistant that turns your PDFs into instant answers. Ask questions, get context-aware responses, and even enjoy a little playful sarcasm along the way. English or Arabic? HAS has you covered.  

---

## 🚀 Features

- **Smart Q&A:** Ask anything about your PDFs and get answers with sources.  
- **Drag & Drop PDFs:** Upload and index PDFs in seconds.  
- **Multilingual:** Supports English & Arabic queries.  
- **Playful Tone:** Friendly, slightly sarcastic, never boring.  
- **Clean UI:** Modern web interface, fully responsive.  
- **Reset Anytime:** Clear uploads and start fresh with one click.  

---

## ⚙️ Tech Stack

- **Backend:** Python + Flask  
- **AI Model:** Mistral 7B (`.gguf`) via `ctransformers`  
- **PDF Handling:** RAG-based context retrieval  
- **Frontend:** HTML, CSS, JS  
- **Translator:** Automatic translation for questions & answers  

---

## 💻 Quick Start

```bash
# Clone the repo
git clone https://github.com/V01Dmlm/HAS.git
cd HAS

# Setup virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux / Mac

# Install dependencies
pip install -r requirements.txt

# Download the model and place in `models/`
# Run the app
python app.py
Open your browser at http://localhost:5000

🎯 Usage
Upload PDFs via drag & drop.

Ask questions naturally in English or Arabic.

Get instant answers, complete with references.

👩‍💻 Contributors
Eng. Samaa

Abdelrhman Wael

Hamdy
