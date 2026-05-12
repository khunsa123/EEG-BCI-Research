"""
agent.py
--------
LangChain AgentExecutor that orchestrates the EEG/BCI Research Copilot.
Routes user queries to the appropriate tools and maintains 
conversation memory across turns.

Usage:
    from src.agent import CopilotAgent
    agent = CopilotAgent()
    response = agent.chat("Summarise the paper eeg_review.pdf")
"""

from pathlib import Path
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage

import sys
sys.path.append(str(Path(__file__).parent))
from config import config
from tools import get_all_tools


AGENT_SYSTEM_PROMPT = """You are an expert EEG and BCI (Brain-Computer Interface) 
research assistant. You help neuroscience and ML researchers understand papers, 
extract methods, compare datasets, design experiments, and generate citations.

You have access to a database of research papers and the following tools:
{tools}

TOOL NAMES: {tool_names}

Use this format EXACTLY when a tool is required:

Question: the problem or request from the user
Thought: think about what to do next
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

RULES:
1. Always start by using list_papers if unsure what papers are available
2. Use answer_question for general EEG/neuroscience questions
3. Use summarize_paper when asked about a specific paper
4. Use extract_methods when asked about techniques or pipelines
5. Use compare_datasets when comparing multiple datasets
6. Use suggest_pipeline when asked to design an analysis approach
7. Use generate_citation when bibliographic information is needed
8. Always cite sources in your Final Answer
9. If a tool returns an error, explain what the user should do
"""


class CopilotAgent:
    """
    Main agent class for the EEG/BCI Research Copilot.
    Wraps a LangChain agent graph with tools and conversation history.
    """

    FALLBACK_MODELS = [
        "gemini-1.5",
        "gemini-1.5-proto",
        "gemini-1.0",
    ]

    def __init__(self):
        """Initialise the agent with LLM, tools, and memory."""
        config.validate()
        print("Initialising EEG/BCI Research Copilot Agent...")

        self._current_model = config.GEMINI_MODEL
        self._tools = get_all_tools()
        self._build_agent(self._current_model)

        self._history: list[HumanMessage | AIMessage] = []
        self._max_history = config.MEMORY_WINDOW

        print("Agent ready. Type your research question.\n")

    def _init_llm(self, model: str):
        self._llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=config.GEMINI_API_KEY,
            temperature=config.GEMINI_TEMPERATURE,
        )
        self._current_model = model
        print(f"✅ Gemini model loaded: {model}")

    def _build_agent(self, model: str):
        self._init_llm(model)
        print(f"Loaded {len(self._tools)} tools: {', '.join(t.name for t in self._tools)}")

        tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in self._tools
        )
        tool_names = ", ".join(tool.name for tool in self._tools)

        system_prompt = AGENT_SYSTEM_PROMPT.format(
            tools=tool_descriptions,
            tool_names=tool_names,
        )

        self._agent = create_agent(
            model=self._llm,
            tools=self._tools,
            system_prompt=system_prompt,
        )

    def chat(self, message: str) -> str:
        """
        Send a message to the agent and get a response.

        Args:
            message: User's research question or request.

        Returns:
            Agent's response string.
        """
        if not message.strip():
            return "⚠️ Please enter a question or request."

        self._history.append(HumanMessage(content=message))
        if len(self._history) > self._max_history * 2:
            self._history = self._history[-self._max_history * 2 :]

        try:
            result = self._agent.invoke({"messages": self._history})

            if isinstance(result, dict):
                output = (
                    result.get("output")
                    or result.get("result")
                    or result.get("text")
                    or result.get("response")
                )
                if not output and "messages" in result:
                    messages = result["messages"]
                    if isinstance(messages, list) and messages:
                        last = messages[-1]
                        output = getattr(last, "content", str(last))
            else:
                output = str(result)

            if not output:
                output = "❌ No response generated."

            self._history.append(AIMessage(content=output))
            return output

        except Exception as e:
            error = str(e)
            if "quota" in error.lower() or "429" in error:
                return (
                    "⚠️ Gemini API rate limit reached. "
                    "Please wait 60 seconds and try again. "
                    "(Free tier allows 15 requests/minute)"
                )
            if "not_found" in error.lower() or "404" in error.lower():
                tried_models = [self._current_model]
                for candidate in self.FALLBACK_MODELS:
                    if candidate == self._current_model:
                        continue
                    tried_models.append(candidate)
                    try:
                        self._build_agent(candidate)
                        result = self._agent.invoke({"messages": self._history})
                        output = (
                            result.get("output")
                            or result.get("result")
                            or result.get("text")
                            or result.get("response")
                        )
                        if not output and "messages" in result:
                            messages = result["messages"]
                            if isinstance(messages, list) and messages:
                                last = messages[-1]
                                output = getattr(last, "content", str(last))
                        if not output:
                            output = "❌ No response generated."
                        self._history.append(AIMessage(content=output))
                        return output
                    except Exception as fallback_error:
                        fallback_msg = str(fallback_error)
                        if "not_found" in fallback_msg.lower() or "404" in fallback_msg.lower():
                            continue
                        error = fallback_msg
                        break
                return (
                    f"❌ Gemini model not available. Tried: {', '.join(tried_models)}. "
                    "Set GEMINI_MODEL in .env to a supported model."
                )
            return f"❌ Agent error: {error}"

    def reset_memory(self) -> None:
        """Clear conversation memory to start fresh."""
        self._history = []
        print("Conversation memory cleared.")

    def get_history(self) -> str:
        """Return current conversation history."""
        if not self._history:
            return "No conversation history yet."
        lines = []
        for message in self._history:
            role = "User" if isinstance(message, HumanMessage) else "Assistant"
            lines.append(f"{role}: {message.content}")
        return "\n".join(lines)


if __name__ == "__main__":
    # CLI mode for quick testing
    print("=" * 60)
    print("EEG/BCI Research Copilot — CLI Mode")
    print("Type 'quit' to exit | 'reset' to clear memory")
    print("=" * 60)

    agent = CopilotAgent()

    while True:
        try:
            user_input = input("\n🔬 You: ").strip()

            if not user_input:
                continue
            elif user_input.lower() == "quit":
                print("👋 Goodbye!")
                break
            elif user_input.lower() == "reset":
                agent.reset_memory()
                continue

            print("\nCopilot:")
            response = agent.chat(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
