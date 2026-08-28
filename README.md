<div align="center">

# 🎬 IMDB Sentiment Analyzer

### A PyTorch RNN + FastAPI + Vanilla JS app that predicts whether a movie review is positive or negative — in real time.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

<img src="https://via.placeholder.com/800x420/0f172a/e2e8f0?text=App+Screenshot+Here" alt="App screenshot" width="700"/>

*Replace the image above with an actual screenshot of your running app.*

</div>

---

## ✨ Overview

This project trains a **Recurrent Neural Network (RNN)** on the IMDB Movie Reviews dataset to classify reviews as **positive** or **negative**, and serves it through a **FastAPI** backend with a clean, dependency-free **HTML/CSS/vanilla JS** frontend.

No React, no build tools, no fuss — clone it, install, run, and you have a working sentiment analyzer in your browser.

## 🚀 Features

- 🧠 **RNN-based sentiment classifier** trained on 50,000 IMDB reviews
- ⚡ **FastAPI backend** with a simple, documented REST endpoint
- 🎨 **Minimal, responsive UI** — no frameworks, just HTML/CSS/JS
- 🧹 **Full NLP preprocessing pipeline** (URL stripping, punctuation removal, HTML tag removal, stopword removal, stemming) applied identically at training and inference time
- 📊 Confidence score displayed as a visual positive/negative gauge
- 🔌 Easy to extend — swap in your own model, dataset, or UI

## 🏗️ Architecture

```
┌─────────────────┐        POST /api/predict        ┌──────────────────────┐
│   Browser (UI)   │ ───────────────────────────────▶│    FastAPI Backend   │
│   index.html     │                                  │    main.py                │
│   style.css      │ ◀─────────────────────────────── │                     │
│   script.js      │      { sentiment, prob }          │  preprocessing.py        │
└─────────────────┘                                  │  model.py               │
                                                       │        ↓                 │
                                                       │  TF-IDF Vectorizer       │
                                                       │        ↓                 │
                                                       │  SentimentRNN (PyTorch)  │
                                                       └──────────────────────┘
```

**Pipeline:** raw text → lowercase → strip URLs → strip punctuation → strip HTML → remove stopwords → stem → TF-IDF vectorize (5000 features) → RNN → sigmoid → `positive` / `negative`

## 📁 Project Structure

```
sentiment-app/
├── backend/
│   ├── main.py                  # FastAPI app & /api/predict endpoint
│   ├── model.py                 # SentimentRNN architecture
│   ├── preprocessing.py         # Text cleaning pipeline
│   ├── requirements.txt
|.  ├── data_preprocessing_and_model_training.ipynb       
│   └── artifacts/               # Trained model + vectorizer (not committed — see below)
│       ├── rnn_sentiment.pth
│       ├── tfidf_vectorizer.pkl
│       └── label_encoder.pkl
├── frontend/
│   ├── index.html               # UI markup
│   ├── style.css                # Styling
│   └── script.js                # Fetch calls + DOM updates
└── README.md
```

## 🧰 Tech Stack

| Layer                | Technology                               |
|-----------------------|-------------------------------------------|
| Model                  | PyTorch (`nn.RNN`)                        |
| Feature engineering    | scikit-learn `TfidfVectorizer`            |
| Backend                | FastAPI + Uvicorn                         |
| Frontend               | HTML5, CSS3, Vanilla JavaScript           |
| NLP toolkit            | NLTK (tokenization, stopwords, stemming)  |

## ⚙️ Getting Started

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/<zakir-maswani>/<Movie-Sentiment-Analysis-with-RNN>.git
cd <your-repo>/sentiment-app
```

### 2. Set up the backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> First run downloads a few small NLTK corpora (`punkt`, `stopwords`) automatically — cached after that.

### 3. Add your trained model artifacts

Make sure `backend/artifacts/` contains:

- `rnn_sentiment.pth`
- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`

### 4. Run the app

```bash
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser 🎉

## 🔌 API Reference

### `POST /api/predict`

Predicts the sentiment of a movie review.

**Request body:**
```json
{
  "review": "This movie was absolutely fantastic, I loved every minute of it!"
}
```

**Response:**
```json
{
  "sentiment": "positive",
  "positive_probability": 0.8734,
  "confidence": 0.8734
}
```

### `GET /api/health`

Simple health check — returns backend status and inference device (`cpu`/`cuda`).

```json
{ "status": "ok", "device": "cpu" }
```

## 🖼️ Screenshots

<div align="center">
<img src="https://via.placeholder.com/700x400/1e293b/e2e8f0?text=Add+a+real+screenshot" width="600"/>
</div>

## 🧪 Model Details

- **Architecture:** Single-layer `nn.RNN` (hidden size 128) → fully connected layer → sigmoid
- **Input representation:** Each review is vectorized with TF-IDF (top 5,000 terms) and fed as a single-timestep sequence
- **Loss / Optimizer:** Binary Cross-Entropy + Adam
- **Dataset:** [IMDB 50K Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/)

> ⚠️ **Note:** Because TF-IDF discards word order before the RNN sees it, the model isn't really leveraging sequential information. For higher accuracy, consider a word-embedding + token-sequence RNN/LSTM as a future improvement.

## 🗺️ Roadmap

- [ ] Replace TF-IDF + single-timestep RNN with word embeddings + true sequence modeling (LSTM/GRU)
- [ ] Add batch prediction endpoint
- [ ] Dockerize the app
- [ ] Add unit tests for preprocessing & API
- [ ] Deploy live demo (Render / Railway / Hugging Face Spaces)

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgements

- [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [PyTorch](https://pytorch.org/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

<div align="center">
Made with ❤️ using PyTorch & FastAPI
</div>
