# 🧠 EEG/BCI Research Copilot

> An AI-powered research assistant for EEG and Brain-Computer Interface literature.
> Upload research papers, then ask questions, extract methods, compare datasets,
> get pipeline suggestions, and generate citations — all grounded in your own paper library.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khunsa123/eeg-bci-copilot/blob/main/notebooks/demo.ipynb)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system with an **agentic workflow** tailored for neuroscience and EEG/BCI research. It demonstrates:

- End-to-end RAG pipeline with local vector storage
- Multi-tool LangChain agent with specialised research capabilities
- Local embeddings (no API cost for semantic search)
- Google Gemini API integration (free tier)
- Gradio web interface deployable locally or on Colab

**Research motivation:** EEG and BCI research involves large volumes of technical literature with specialised terminology. Researchers spend significant time manually searching papers for specific preprocessing methods, dataset characteristics, and architectural choices. This tool automates that process while grounding answers in cited sources.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Gradio Web UI                         │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              LangChain ReAct Agent                       │
│         (Google Gemini 1.5 Flash — free tier)            │
└────┬──────┬──────┬──────┬──────┬──────┬────────────────-┘
     │      │      │      │      │      │
  list   summ  extract compare suggest  cite
 papers  paper methods datasets pipeline citation
     │      │      │      │      │      │
┌────▼──────▼──────▼──────▼──────▼──────▼────────────────┐
│                   Retriever                              │
│     sentence-transformers/all-MiniLM-L6-v2 (local)      │
│                  ChromaDB (local)                        │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   Ingest Pipeline                        │
│         PyMuPDF → Text Splitter → Embeddings             │
│                  PDF Research Papers                     │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| PDF parsing | PyMuPDF | Free |
| Text splitting | LangChain RecursiveCharacterTextSplitter | Free |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Free (local) |
| Vector store | ChromaDB (persistent, local) | Free |
| LLM | Google Gemini 1.5 Flash | Free (15 req/min) |
| Agent | LangChain ReAct | Free |
| UI | Gradio | Free |

**Total cost: $0** — runs entirely on free tier services.

> Tip: The app includes fallback logic for Gemini compatibility. If `gemini-1.5-flash` is unavailable, set `GEMINI_MODEL=gemini-1.5` or `gemini-1.5-proto` in `.env`.

---

## 🚀 Quick Start

### Option A: Run Locally

```bash
# 1. Clone repository
git clone https://github.com/khunsa123/eeg-bci-copilot.git
cd eeg-bci-copilot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API key
cp .env.example .env
# Edit .env and add your Gemini API key
# Get free key at: https://aistudio.google.com

# 5. Add your papers
cp your_papers/*.pdf data/papers/

# 6. Ingest papers
python src/ingest.py

# 7. Launch app
python app.py
# Open: http://localhost:7860
```

### Option B: Run in Google Colab
Click the **Open in Colab** badge above and follow the notebook steps.

---

## 🔧 Agent Tools

| Tool | Description | Example Query |
|------|-------------|--------------|
| `list_papers` | Show all ingested papers | *"What papers do you have?"* |
| `summarize_paper` | Structured paper summary | *"Summarise eeg_review.pdf"* |
| `extract_methods` | Extract preprocessing & ML methods | *"What ICA methods are used?"* |
| `compare_datasets` | Dataset comparison table | *"Compare DEAP and MAHNOB"* |
| `suggest_pipeline` | Recommend analysis pipeline | *"Design a pipeline for emotion recognition"* |
| `generate_citation` | APA + BibTeX citations | *"Cite eeg_review.pdf"* |
| `answer_question` | General Q&A | *"What is phase-amplitude coupling?"* |

---

## 📂 Project Structure

```
eeg-bci-copilot/
│
├── app.py                    # Gradio web interface
│
├── src/
│   ├── config.py             # Central configuration
│   ├── ingest.py             # PDF ingestion pipeline
│   ├── retriever.py          # Semantic search
│   ├── generator.py          # Gemini API generation
│   ├── tools.py              # LangChain agent tools
│   └── agent.py              # AgentExecutor orchestrator
│
├── notebooks/
│   └── demo.ipynb            # Google Colab demo
│
├── data/
│   ├── papers/               # Add your PDFs here (gitignored)
│   └── chroma_db/            # Vector store (gitignored)
│
├── requirements.txt
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## ⚙️ Configuration

All settings are in `src/config.py` and can be overridden via `.env`:

```bash
GEMINI_API_KEY=your_key_here
CHUNK_SIZE=500          # Tokens per chunk
CHUNK_OVERLAP=50        # Overlap between chunks
TOP_K_RESULTS=5         # Chunks retrieved per query
COLLECTION_NAME=eeg_papers
```

---

## 📊 Example Session

```
🔬 You: List all papers in the database

🤖 Copilot: 📚 3 paper(s) in database:
  1. eeg_emotion_recognition.pdf
  2. bci_motor_imagery.pdf
  3. eeg_preprocessing_review.pdf

🔬 You: Compare the DEAP and MAHNOB-HCI datasets

🤖 Copilot: | Feature | DEAP | MAHNOB-HCI |
| Subjects | 32 | 27 |
| EEG Channels | 32 | 32 |
| Labels | Valence, Arousal, Dominance | Valence, Arousal |
| Stimuli | Music videos (1 min) | Movie clips (35s-117s) |
| Sampling Rate | 512 Hz (downsampled to 128) | 256 Hz |
[Source: eeg_emotion_recognition.pdf, Page 3]

🔬 You: Suggest a pipeline for attention classification

🤖 Copilot: ## Recommended Pipeline for EEG Attention Classification
### Preprocessing
1. Bandpass filter: 1–40 Hz (Butterworth 4th order)
2. Notch filter: 50 Hz
3. ICA: remove ocular and muscular artifacts
...
```

---

## 🔗 Related Work

This project is part of a broader research portfolio:

- **Published:** Ahmed, W., Riaz, S., Iftikhar, K., Konur, S. (2023). *Speech Emotion Recognition Using Deep Learning.* Springer LNCS Vol. 14381, SGAI 2023. [DOI: 10.1007/978-3-031-47994-6_14](https://doi.org/10.1007/978-3-031-47994-6_14)
- **In preparation:** Iftikhar, K., Nisar, M.W. *EEG-Based Attention and Cognitive State Analysis Using Consumer-Grade BCI Devices.*
- **Related repo:** [Multimodal-NeuroPhysio-Signal-Research](https://github.com/khunsa123/Multimodal-NeuroPhysio-Signal-Research) — EEG signal classification pipelines

---

## 👩‍🔬 Author

**Khunsa Iftikhar**  
Computational Neuroscience & AI Researcher  
🔗 [GitHub](https://github.com/khunsa123) | [Google Scholar](https://scholar.google.com/citations?hl=en&user=Q-mM508AAAAJ)

📬 khunsaiftikhar123@gmail.com

---

## ⚠️ Notes

- **API rate limits:** Gemini free tier allows 15 requests/minute. If you hit limits, wait 60 seconds.
- **Privacy:** Papers are stored locally in ChromaDB. Nothing is sent to external servers except the text chunks sent to Gemini for generation.
- **Paper size:** Very large PDFs (>100 pages) may take a few minutes to ingest.
- **License:** This project uses Apache License 2.0, which allows commercial use while requiring attribution and maintaining patent protections. This supports potential future publication and commercialization of the research.
