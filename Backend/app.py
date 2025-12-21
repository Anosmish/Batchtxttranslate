from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os, uuid, shutil, zipfile, time, threading
import nltk
from googletrans import Translator

nltk.download('punkt')

app = Flask(__name__)
CORS(app)

# ---------------- CONFIG ----------------
BATCH_WORD_LIMIT = 9000
BASE_OUTPUT = "output"
PROJECT_EXPIRY = 30 * 60  # 30 minutes in seconds
os.makedirs(BASE_OUTPUT, exist_ok=True)

# ---------------- HELPERS ----------------
def split_into_sentences(text):
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)

def create_batches(sentences, word_limit):
    batches, current_batch, current_count = [], [], 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if current_count + word_count > word_limit:
            batches.append(" ".join(current_batch))
            current_batch, current_count = [sentence], word_count
        else:
            current_batch.append(sentence)
            current_count += word_count
    if current_batch:
        batches.append(" ".join(current_batch))
    return batches

def translate_batch(text, target_lang):
    translator = Translator()
    return translator.translate(text, dest=target_lang).text

def zip_folder(folder_path, zip_name):
    shutil.make_archive(zip_name.replace(".zip",""), 'zip', folder_path)
    return zip_name

def cleanup_old_projects():
    while True:
        try:
            now = time.time()
            for folder in os.listdir(BASE_OUTPUT):
                path = os.path.join(BASE_OUTPUT, folder)
                if os.path.isdir(path) or path.endswith(".zip"):
                    mtime = os.path.getmtime(path)
                    if now - mtime > PROJECT_EXPIRY:
                        try:
                            if os.path.isdir(path):
                                shutil.rmtree(path)
                            else:
                                os.remove(path)
                        except:
                            pass
        except Exception as e:
            print("Cleanup Error:", e)
        time.sleep(60)  # check every 1 min

# ---------------- ROUTES ----------------
@app.route("/translate", methods=["POST"])
def translate_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    target_lang = request.form.get("language", "hi")

    text = file.read().decode("utf-8")
    sentences = split_into_sentences(text)
    batches = create_batches(sentences, BATCH_WORD_LIMIT)

    project_id = str(uuid.uuid4())
    project_folder = os.path.join(BASE_OUTPUT, f"project_{project_id}")
    os.makedirs(project_folder, exist_ok=True)

    # Translate and save batches
    for idx, batch in enumerate(batches, start=1):
        translated_text = translate_batch(batch, target_lang)
        batch_file = os.path.join(project_folder, f"{idx}.txt")
        with open(batch_file, "w", encoding="utf-8") as f:
            f.write(translated_text)

    # Zip folder
    zip_name = os.path.join(BASE_OUTPUT, f"project_{project_id}.zip")
    zip_folder(project_folder, zip_name)

    # Keep project folder for history for 30 min, cleanup thread will remove later

    return jsonify({"download_url": f"/download/{os.path.basename(zip_name)}"})

@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    file_path = os.path.join(BASE_OUTPUT, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_projects, daemon=True)
    cleanup_thread.start()

    app.run(host="0.0.0.0", port=5000, debug=True)
