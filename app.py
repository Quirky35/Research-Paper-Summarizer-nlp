from flask import Flask, render_template, request, send_from_directory, make_response
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pdfplumber
import re
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Load DistilBART model on GPU
# -----------------------------
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# Utility Functions
# -----------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def summarize_text(text, max_length=400, min_length=100):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        min_length=min_length,
        length_penalty=1.5,
        num_beams=6,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_large_text(text, chunk_size=800):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) < chunk_size:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    if current_chunk:
        chunks.append(current_chunk.strip())

    full_summary = ""
    for i, chunk in enumerate(chunks):
        summary = summarize_text(chunk)
        summary = remove_numbers(summary)  # ⬅️ Remove numbers only from the summary
        full_summary += f"\n\n🔹 Part {i+1}:\n{summary}"

    return full_summary.strip()

# -----------------------------
# Flask Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'pdf_file' not in request.files:
        return "No file uploaded", 400

    file = request.files['pdf_file']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Extract and summarize
    extracted_text = extract_text_from_pdf(filepath)
    cleaned_text = clean_text(extracted_text)
    summary = summarize_large_text(cleaned_text)

    # Save summary as .txt
    summary_filename = f"{os.path.splitext(filename)[0]}_summary.txt"
    summary_path = os.path.join(UPLOAD_FOLDER, summary_filename)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)

    return render_template('result.html', summary=summary, summary_filename=summary_filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
