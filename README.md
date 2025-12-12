Research Paper Summarizer (NLP + Flask Web App)

A lightweight and powerful AI-based document summarizer built using Transformers, Hugging Face, and Flask.
This web application allows users to upload research papers, articles, reports, or any long text and automatically generates a concise summary using state-of-the-art NLP models.

🚀 Features

📄 Upload research papers or text files
🤖 Uses BART Large CNN (Hugging Face) for high-quality abstractive summarization
⚡ Clean, simple, responsive Flask web UI
📦 No huge model files stored in repo — model loads automatically from Hugging Face
🔍 Extracts key information from long documents
🧠 Ideal for students, researchers, analysts, and developers
📝 Easy to extend (can replace model with Pegasus, T5, Longformer, etc.)
🧠 Model Used

This project uses:

facebook/bart-large-cnn
A transformer-based sequence-to-sequence model designed for abstractive summarization.
The model is downloaded automatically at runtime:
from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
No manual setup required.

🛠️ Tech Stack

Component	Technology
Backend	Flask
NLP Model	Hugging Face Transformers
Language	Python
Frontend	HTML, CSS
File Handling	werkzeug / Flask

📦 Installation & Setup

1️⃣ Clone the repository
git clone https://github.com/Quirky35/Research-Paper-Summarizer-nlp.git
cd Research-Paper-Summarizer-nlp

2️⃣ Create & activate a virtual environment
python -m venv flask_env
flask_env\Scripts\activate   # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

If you don’t have a requirements.txt, here is a minimal version:

flask
transformers
torch

4️⃣ Run the Flask app
python app.py

Open in browser:

http://127.0.0.1:5000

🖼️ Usage
Open the app in your browser

Upload a PDF, text file, or paste text directly
Click Summarize
Wait a few seconds while the model processes the tex
A clean summarized output will appear on the result page

📁 Project Structure
├── app.py                    # Main Flask server
├── summarizer.py             # Summarization logic
├── templates/
│   ├── index.html            # Upload UI
│   └── result.html           # Summary output page
├── static/
│   └── styles.css            # Styling
└── .gitignore                # Ignore large models & environments

📌 Why This Project Is Useful

Saves hours of reading time
Helps students quickly understand research papers
Ideal for analyzing long documents
Showcases expertise in Python, NLP, Web Dev, and Deep Learning
Great addition to resume & portfolio

🚀 Future Enhancements
✔ Support for large PDFs (chunking + long transformers)
✔ Add keywords extraction
✔ Add multilingual summarization
✔ Add summary length control
✔ Deploy on Render / HuggingFace Spaces
If you want help implementing any of these, just ask!

🤝 Contributions
Pull requests are welcome!
Feel free to open issues or request new features.

⭐ Show Support

If you found this helpful, please ⭐ the repo to support the project.
