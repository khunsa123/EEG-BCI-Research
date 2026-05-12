"""
generator.py
------------
Handles response generation using Google Gemini API.
Takes retrieved context + user query and generates
a structured, cited research response.

Usage:
    from src.generator import Generator
    gen = Generator()
    response = gen.generate(query="What is ICA?", context="...")
"""

from pathlib import Path
from typing import List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

import sys
sys.path.append(str(Path(__file__).parent))
from config import config


class Generator:
    """
    Response generator using Google Gemini API.
    Uses the configured model and falls back to alternatives if needed.
    """

    FALLBACK_MODELS = [
        "gemini-1.5",
        "gemini-1.5-proto",
        "gemini-1.0",
    ]

    def __init__(self):
        """Initialise Gemini LLM."""
        config.validate()
        self._current_model = config.GEMINI_MODEL
        self._init_llm(self._current_model)

    def _init_llm(self, model: str):
        self._llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=config.GEMINI_API_KEY,
            temperature=config.GEMINI_TEMPERATURE,
            max_output_tokens=config.GEMINI_MAX_TOKENS,
        )
        self._current_model = model
        print(f"✅ Generator ready. Model: {model}")

    def generate(
        self,
        query: str,
        context: str,
        task_type: str = "qa",
    ) -> str:
        """
        Generate a response to a query given retrieved context.

        Args:
            query: The user's question or request.
            context: Retrieved paper chunks as formatted string.
            task_type: Type of task — affects prompt framing.
                Options: "qa", "summarise", "compare", 
                         "pipeline", "citation"

        Returns:
            Generated response string.
        """
        # Build task-specific prompt
        task_prompts = {
            "qa": self._qa_prompt(query, context),
            "summarise": self._summarise_prompt(query, context),
            "compare": self._compare_prompt(query, context),
            "pipeline": self._pipeline_prompt(query, context),
            "citation": self._citation_prompt(query, context),
        }

        prompt = task_prompts.get(task_type, task_prompts["qa"])

        try:
            messages = [
                SystemMessage(content=config.SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = self._llm.invoke(messages)
            return response.content

        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "429" in error_msg:
                return (
                    "⚠️ Gemini API rate limit reached. "
                    "Wait 60 seconds and try again. "
                    "(Free tier: 15 requests/minute)"
                )
            elif "api_key" in error_msg.lower():
                return (
                    "❌ Invalid Gemini API key. "
                    "Check your .env file."
                )
            elif "not_found" in error_msg.lower() or "404" in error_msg:
                tried_models = [self._current_model]
                for candidate in self.FALLBACK_MODELS:
                    if candidate == self._current_model:
                        continue
                    try:
                        self._init_llm(candidate)
                        response = self._llm.invoke(messages)
                        return response.content
                    except Exception as fallback_error:
                        fallback_msg = str(fallback_error)
                        tried_models.append(candidate)
                        if "not_found" in fallback_msg.lower() or "404" in fallback_msg:
                            continue
                        error_msg = fallback_msg
                        break
                return (
                    f"❌ Model not available. Tried: {', '.join(tried_models)}. "
                    "Set GEMINI_MODEL in .env to a supported model."
                )
            else:
                return f"❌ Generation error: {error_msg}"

    def _qa_prompt(self, query: str, context: str) -> str:
        """Prompt for general question answering."""
        return f"""Based on the following research paper excerpts, 
answer the question precisely and scientifically.

CONTEXT FROM PAPERS:
{context}

QUESTION: {query}

Instructions:
- Answer based only on the provided context
- Cite specific papers and page numbers for key claims
- Use precise scientific terminology
- If context is insufficient, state this clearly
- Structure your answer clearly with key points"""

    def _summarise_prompt(self, query: str, context: str) -> str:
        """Prompt for paper summarisation."""
        return f"""Summarise the following research paper content 
in a structured academic format.

PAPER CONTENT:
{context}

Paper/Topic to summarise: {query}

Provide a structured summary with these sections:
## Research Objective
## Methods & Dataset
## Key Findings
## Limitations
## Relevance to EEG/BCI Research"""

    def _compare_prompt(self, query: str, context: str) -> str:
        """Prompt for dataset/method comparison."""
        return f"""Compare the datasets or methods mentioned 
in the following context.

CONTEXT:
{context}

Comparison request: {query}

Format your response as a structured comparison table using 
markdown. Include: name, key characteristics, advantages, 
limitations, and typical use cases."""

    def _pipeline_prompt(self, query: str, context: str) -> str:
        """Prompt for preprocessing pipeline suggestion."""
        return f"""Based on the methods described in the following 
research papers, suggest a preprocessing and ML pipeline.

CONTEXT FROM PAPERS:
{context}

Research question/task: {query}

Provide a detailed pipeline with these sections:
## Recommended Preprocessing Steps
(list each step with justification from the papers)

## Feature Extraction Strategy
(which features to extract and why)

## Suggested ML Architecture
(model type with rationale)

## Evaluation Protocol
(validation strategy, metrics)

## Implementation Notes
(practical tips and potential pitfalls)

Cite specific papers for each major recommendation."""

    def _citation_prompt(self, query: str, context: str) -> str:
        """Prompt for citation generation."""
        return f"""Extract bibliographic information from the 
following paper content and generate proper citations.

PAPER CONTENT:
{context}

Paper to cite: {query}

Generate citations in these formats:

## APA Format
[generate APA citation]

## BibTeX Format
```bibtex
[generate BibTeX entry]
```

## Key Information Extracted
- Title:
- Authors:
- Year:
- Venue/Journal:
- DOI (if found):
- Volume/Pages (if found):"""


if __name__ == "__main__":
    # Quick test — requires valid API key in .env
    gen = Generator()
    test_response = gen.generate(
        query="What is independent component analysis in EEG?",
        context="ICA is a signal processing method used to separate "
                "mixed signals into independent components. In EEG, "
                "it removes artifacts like eye blinks and muscle noise. "
                "(Source: test_paper.pdf, Page 3)",
        task_type="qa"
    )
    print("\n📝 Test response:")
    print(test_response)
