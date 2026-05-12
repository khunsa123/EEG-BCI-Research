"""
app.py
------
Gradio web interface for the EEG/BCI Research Copilot.
Provides:
  - PDF upload and ingestion
  - Chat interface
  - Source/context display panel
  - Paper library view

Run locally:    python app.py
Run in Colab:   python app.py --share  (generates public URL)
"""

import argparse
from pathlib import Path
import gradio as gr

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
sys.path.append(str(Path(__file__).parent / "src"))
from config import config
from ingest import ingest_pdfs
from retriever import Retriever
from agent import CopilotAgent


# ── Initialise components ─────────────────────────────────────────────────────
print("Starting EEG/BCI Research Copilot...")
config.validate()

retriever = None  # Lazy load the retriever
agent = None  # Lazy load the agent

def get_retriever():
    global retriever
    if retriever is None:
        retriever = Retriever()
    return retriever

def get_agent():
    global agent
    if agent is None:
        agent = CopilotAgent()
    return agent


# ── Event handlers ────────────────────────────────────────────────────────────

def upload_papers(files) -> str:
    """
    Handle PDF file uploads. Ingests each file into ChromaDB.

    Args:
        files: List of uploaded file objects from Gradio.

    Returns:
        Status message string.
    """
    if not files:
        return "⚠️ No files uploaded."

    results = []
    for file in files:
        file_path = file.name
        filename = Path(file_path).name
        chunks = ingest_pdfs(specific_file=file_path)
        if chunks > 0:
            results.append(f"✅ {filename} — {chunks} chunks ingested")
        else:
            results.append(f"⚠️ {filename} — already in database or empty")

    return "\n".join(results)


def list_papers_ui() -> str:
    """Return formatted list of papers for display."""
    papers = get_retriever().list_papers()
    if not papers:
        return "📭 No papers in database. Upload PDFs above."
    lines = [f"{i+1}. {p}" for i, p in enumerate(papers)]
    return f"📚 {len(papers)} paper(s) in database:\n" + "\n".join(lines)


def _normalize_history(history: list) -> list:
    normalized_history = []
    if not history:
        return normalized_history

    for message in history:
        if isinstance(message, dict) and "role" in message and "content" in message:
            normalized_history.append(message)
        elif isinstance(message, (list, tuple)) and len(message) == 2:
            normalized_history.append({"role": "user", "content": message[0]})
            normalized_history.append({"role": "assistant", "content": message[1]})
    return normalized_history


def chat(
    message: str,
    history: list,
    show_sources: bool,
) -> tuple:
    """
    Handle chat message. Returns updated history and sources.

    Args:
        message: User's input message.
        history: Gradio chat history list.
        show_sources: Whether to display retrieved sources.

    Returns:
        Tuple of (updated_history, sources_text, empty_input)
    """
    if not message.strip():
        return history, "", ""

    history = _normalize_history(history)

    # Get agent response
    response = get_agent().chat(message)

    # Get retrieved sources for display
    sources_text = ""
    if show_sources:
        search_retriever = get_retriever()
        results = search_retriever.search(query=message, k=3)
        if results:
            source_lines = []
            for i, r in enumerate(results, 1):
                source_lines.append(
                    f"**[{i}] {r['filename']} — Page {r['page']}** "
                    f"(score: {r['score']})\n"
                    f"_{r['text'][:300]}..._"
                )
            sources_text = "\n\n---\n\n".join(source_lines)
        else:
            sources_text = "No sources retrieved for this query."

    # Update history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, sources_text, ""


def reset_chat() -> tuple:
    """Reset chat history and memory."""
    get_agent().reset_memory()
    return [], "💬 Chat reset. Memory cleared.", ""


# ── Example queries ───────────────────────────────────────────────────────────
EXAMPLE_QUERIES = [
    "List all papers in the database",
    "What EEG preprocessing steps are recommended for emotion recognition?",
    "Compare the DEAP and MAHNOB-HCI datasets",
    "Suggest a pipeline for EEG-based attention classification",
    "What CNN architectures are used for EEG classification?",
    "What is the best ICA method for EEG artifact removal?",
    "Generate a citation for [paper_name].pdf",
    "Summarise the paper [paper_name].pdf",
]


# ── Build Gradio interface ────────────────────────────────────────────────────
with gr.Blocks(
    title="EEG/BCI Research Copilot",
) as demo:

    gr.Markdown("""
    # 🧠 EEG/BCI Research Copilot
    **AI-powered research assistant for EEG and Brain-Computer Interface literature.**
    
    Upload your research papers, then ask questions, extract methods, 
    compare datasets, get pipeline suggestions, and generate citations.
    
    *Powered by: Google Gemini API + ChromaDB + LangChain + Sentence Transformers*
    """)

    with gr.Tabs():

        # ── Tab 1: Chat ───────────────────────────────────────────────────────
        with gr.Tab("💬 Research Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="Research Assistant",
                        height=500,
                        show_label=True,
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask about EEG methods, datasets, "
                                        "papers, or request a pipeline...",
                            label="Your question",
                            lines=2,
                            scale=4,
                        )
                        submit_btn = gr.Button("Send 🚀", scale=1, variant="primary")

                    with gr.Row():
                        reset_btn = gr.Button("🗑️ Reset Chat", variant="secondary")
                        show_sources = gr.Checkbox(
                            label="Show retrieved sources",
                            value=True,
                        )

                    gr.Examples(
                        examples=EXAMPLE_QUERIES,
                        inputs=msg_input,
                        label="Example queries (click to use):",
                    )

                with gr.Column(scale=1):
                    sources_display = gr.Markdown(
                        label="📄 Retrieved Sources",
                        value="*Sources will appear here after each query.*"
                    )

            # Wire up events
            submit_btn.click(
                fn=chat,
                inputs=[msg_input, chatbot, show_sources],
                outputs=[chatbot, sources_display, msg_input],
            )
            msg_input.submit(
                fn=chat,
                inputs=[msg_input, chatbot, show_sources],
                outputs=[chatbot, sources_display, msg_input],
            )
            reset_btn.click(
                fn=reset_chat,
                outputs=[chatbot, sources_display, msg_input],
            )

        # ── Tab 2: Upload Papers ──────────────────────────────────────────────
        with gr.Tab("📤 Upload Papers"):
            gr.Markdown("""
            ### Upload EEG/BCI Research Papers
            Upload PDF files to add them to the knowledge base.
            Files are processed locally — your papers stay private.
            """)

            with gr.Row():
                with gr.Column():
                    file_upload = gr.File(
                        label="Upload PDF papers",
                        file_types=[".pdf"],
                        file_count="multiple",
                    )
                    upload_btn = gr.Button(
                        "📥 Ingest Papers", variant="primary"
                    )
                    upload_status = gr.Textbox(
                        label="Ingestion Status",
                        lines=5,
                        interactive=False,
                    )

                with gr.Column():
                    papers_display = gr.Textbox(
                        label="Papers in Database",
                        lines=15,
                        interactive=False,
                        value=list_papers_ui(),
                    )
                    refresh_btn = gr.Button("🔄 Refresh List")

            upload_btn.click(
                fn=upload_papers,
                inputs=file_upload,
                outputs=upload_status,
            )
            refresh_btn.click(
                fn=list_papers_ui,
                outputs=papers_display,
            )
            upload_btn.click(
                fn=list_papers_ui,
                outputs=papers_display,
            )

        # ── Tab 3: About ──────────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown(f"""
            ## About EEG/BCI Research Copilot
            
            This tool was built as part of a research portfolio project 
            demonstrating RAG (Retrieval-Augmented Generation) and 
            agentic AI applied to neuroscience research.
            
            ### Architecture
            - **Document Processing:** PyMuPDF → LangChain text splitter
            - **Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (local)
            - **Vector Store:** ChromaDB (persistent, local)
            - **LLM:** Google Gemini 1.5 Flash (free tier)
            - **Agent:** LangChain ReAct agent with 7 specialised tools
            - **UI:** Gradio
            
            ### Available Tools
            | Tool | Description |
            |------|-------------|
            | list_papers | Show all papers in database |
            | summarize_paper | Structured summary of a paper |
            | extract_methods | Extract preprocessing & ML methods |
            | compare_datasets | Dataset comparison table |
            | suggest_pipeline | Recommend analysis pipeline |
            | generate_citation | APA + BibTeX citations |
            | answer_question | General Q&A |
            
            ### Configuration
            - **Embedding Model:** `{config.EMBEDDING_MODEL}`
            - **LLM Model:** `{config.GEMINI_MODEL}`
            - **Chunk Size:** {config.CHUNK_SIZE} tokens
            - **Top-K Retrieval:** {config.TOP_K_RESULTS} chunks
            - **Memory Window:** {config.MEMORY_WINDOW} turns
            
            ### Author
            **Khunsa Iftikhar**  
            github.com/khunsa123
            """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--share", action="store_true",
        help="Generate public Gradio URL (needed for Colab)"
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Local port to run on (default: 7860)"
    )
    args = parser.parse_args()

    demo.launch(
        share=args.share,          # True for Colab, False for local
        server_port=args.port,
        show_error=True,
        theme=gr.themes.Soft(),
        css=".gradio-container {max-width: 1200px; margin: auto}",
    )
