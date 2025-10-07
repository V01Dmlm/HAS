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
1. Clone the Repository
```bash
git clone https://github.com/V01Dmlm/HAS.git
cd HAS
 ```
2. Run the Setup Script

We provide a script that creates a virtual environment, installs dependencies, and downloads the Mistral model automatically.

Windows:
```bash
setup.bat
 ```

Linux / Mac:
```bash
chmod +x setup.sh
./setup.sh
 ```

The script will:

Create a .venv virtual environment

Activate it

Install all dependencies from requirements.txt

Download the Mistral 7B Instruct model into models/

3. Run the App
```bash
python app.py
 ```

Open your browser at:
```bash
http://localhost:5000
 ```
🎯 Usage
Upload PDFs via drag & drop.

Ask questions naturally in English or Arabic.

Get instant answers, complete with references.

👩‍💻 Contributors

Eng. Samaa

Abdelrhman Wael

Hamdy faheem
