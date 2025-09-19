# app.py (Flask + Mistral 7B + PDF RAG + Translator, simplified)
import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pathlib, re

from backend.chatbot import ChatBot
from backend.pdf_handler import PDFHandler
from backend.translator import Translator

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HAS")

# ----------------- Upload Setup -----------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB

# ----------------- Filename Helpers -----------------
def allowed_file(filename: str):
    return pathlib.Path(filename).suffix.lower() == ".pdf"

def sanitize_filename(filename: str):
    return re.sub(r'[\/\\\?\%\*\:\|\"<>\s]', "_", filename)

# ----------------- GPU / Environment -----------------
os.environ["CT_THREADS"] = str(os.cpu_count())
os.environ["CT_USE_CUDA"] = "1"

# ----------------- Initialize Models -----------------
translator = Translator()
chatbot = ChatBot(model_name="mistral-7b-instruct-v0.2.Q4_K_M.gguf")
pdf_handler = PDFHandler(max_workers=6, chunk_size=400, overlap=50)

# ----------------- Flask Setup -----------------
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ----------------- Error Handlers -----------------
@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Server error"}), 500

# ----------------- Routes -----------------
@app.route("/")
def index():
    if os.path.exists("index.html"):
        return send_from_directory(".", "index.html")
    return jsonify({"success": False, "error": "index.html not found"}), 404

@app.route("/upload", methods=["POST"])
def upload_files():
    uploaded_files = request.files.getlist("files")
    saved, errors = [], []

    if not uploaded_files:
        return jsonify({"success": False, "uploaded": [], "errors": ["No files received."]})

    for f in uploaded_files:
        original_name = f.filename
        if not allowed_file(original_name):
            errors.append(f"{original_name}: invalid file type")
            continue

        filename = sanitize_filename(original_name)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        try:
            f.save(file_path)
            logger.info(f"[UPLOAD] Saved file: {file_path}")
            pdf_handler.save_pdf(file_path)
            logger.info(f"[INDEX] Indexed PDF: {original_name}")
            saved.append(original_name)
        except Exception as e:
            msg = f"{original_name}: failed to save or index ({e})"
            logger.error(msg)
            errors.append(msg)

    return jsonify({"success": bool(saved), "uploaded": saved, "errors": errors})

@app.route("/files", methods=["GET"])
def list_files():
    files = []
    for fname in os.listdir(app.config["UPLOAD_FOLDER"]):
        fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        if os.path.isfile(fpath):
            files.append({
                "filename": fname,
                "size": os.path.getsize(fpath),
                "type": pathlib.Path(fname).suffix.lower()[1:]
            })
    return jsonify({"files": files})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True) or {}
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"success": False, "response": "Empty question."})

        # ----------------- Detect user language -----------------
        user_lang = translator.detect_language(question)
        if not user_lang or user_lang not in ["en", "ar"]:
            user_lang = "ar" if re.search(r'[\u0600-\u06FF]', question) else "en"

        # ----------------- Supercharged greetings with sarcasm -----------------
        greetings_ar = ["مرحبا", "أهلا", "سلام", "هاي", "هلا", "أهلاً وسهلاً"]
        greetings_en = ["hi", "hello", "hey", "hiya", "yo", "sup"]
        greetings_all = [g.lower() for g in greetings_ar + greetings_en]

        q_norm = question.lower().strip()
        if q_norm in greetings_all or len(q_norm.split()) <= 2:
            responses_ar = [
                "أهلا! على الأقل لم تضع لي سؤالًا عن الرياضيات 😏",
                "مرحبا! ماذا تريد هذه المرة؟ 😎",
                "أهلا وسهلا! 🌟 لا تقل لي أنك ستسأل عن الطقس!"
            ]
            responses_en = [
                "Hello! Finally, someone greeted me 😏",
                "Hey there! What trouble are we getting into today? 😎",
                "Hi! Don't tell me you're asking about the weather again!"
            ]
            import random
            answer = random.choice(responses_ar if user_lang == "ar" else responses_en)
            return jsonify({"success": True, "response": answer, "sources": []})

        # ----------------- General small-talk with sarcasm -----------------
        small_talk_patterns = [
            r"how are you", r"what's up", r"tell me a joke", r"كيف حالك", r"ماذا تفعل", r"أخبرني بنكتة"
        ]
        for pattern in small_talk_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                responses_ar = [
                    "أنا بخير، على عكس الإنترنت 😏",
                    "كل شيء ممتاز! وأنت؟ لا تكن كسولاً 😎",
                    "أعمل على مساعدة البشر، مثلك 😉"
                ]
                responses_en = [
                    "I'm good, unlike the WiFi 😏",
                    "All is great! How about you? Don't be lazy 😎",
                    "Just helping humans like you 😉"
                ]
                import random
                answer = random.choice(responses_ar if user_lang == "ar" else responses_en)
                return jsonify({"success": True, "response": answer, "sources": []})

        # ----------------- Translate question to English if needed -----------------
        translated_question = translator.translate_to_english(question) if user_lang != "en" else question

        # ----------------- Gather PDF files -----------------
        pdf_files = [f for f in os.listdir(app.config["UPLOAD_FOLDER"]) if allowed_file(f)]

        # ----------------- Get context from PDFs -----------------
        context_data = pdf_handler.get_context(
            translated_question, top_k=5, pdf_files=pdf_files, max_chars=1500
        )
        context_text = context_data.get("text", "") or "No relevant context found in uploaded PDFs."
        sources = context_data.get("sources", [])

        # ----------------- Translate PDF chunks to English if needed -----------------
        pdf_lang = translator.detect_language(context_text)
        if pdf_lang != "en":
            context_text = translator.translate_to_english(context_text)

        # ----------------- Ask the LLM -----------------
        answer = chatbot.ask(translated_question, context_text)

        # ----------------- Translate back if user is not English -----------------
        if user_lang != "en":
            try:
                answer = translator.translate(answer, to_lang=user_lang)
            except Exception as e:
                logger.warning(f"Translation back failed, returning English: {e}")

        return jsonify({"success": True, "response": answer, "sources": sources})

    except Exception as e:
        logger.error(f"Unexpected error in /chat: {e}", exc_info=True)
        return jsonify({"success": False, "response": "Error processing your request."})



@app.route("/reset", methods=["POST"])
def reset():
    for fname in os.listdir(app.config["UPLOAD_FOLDER"]):
        try:
            os.remove(os.path.join(app.config["UPLOAD_FOLDER"], fname))
        except Exception as e:
            logger.warning(f"Failed to remove {fname}: {e}")
    pdf_handler.clear_index()
    return jsonify({"success": True, "message": "Session reset."})

# ----------------- Run Server -----------------
if __name__ == "__main__":
    # Replace 5000 with your ngrok port
    app.run(host="0.0.0.0", port=80, debug=False)
