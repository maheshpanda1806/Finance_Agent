# ------------------------------------------------------------
# 🧠 Multi-Agent System using Phi, Groq, and external data tools
# ------------------------------------------------------------
# This script demonstrates how to build specialized AI agents that:
# 1. Use different data sources (like DuckDuckGo, Yahoo Finance)
# 2. Collaborate under a parent “multi-agent” to combine insights
# 3. Produce structured, source-backed responses
# ------------------------------------------------------------

# --- Import necessary libraries and tools ---
from phi.agent import Agent                     # Base Agent class to create AI agents
from phi.model.groq import Groq                 # Groq model wrapper for large language models (LLMs)
from phi.tools.yfinance import YFinanceTools    # Financial data tool for stock analysis and news
from phi.tools.duckduckgo import DuckDuckGo     # Web search tool to fetch recent information from the web
from dotenv import load_dotenv                  # For securely loading environment variables from a .env file
import os                                       # Provides access to environment variables and file paths


# --- Step 1: Load environment variables from .env file ---
# The .env file typically contains sensitive data like API keys.
# Example: GROQ_API_KEY=your_api_key_here
load_dotenv()

# --- Step 2: Print sanity check to confirm environment setup ---
# These print statements help verify that the necessary environment variables are properly loaded.
print("✅ GROQ_API_KEY loaded:", bool(os.getenv("GROQ_API_KEY")))
print("✅ PHI_DEFAULT_MODEL:", os.getenv("PHI_DEFAULT_MODEL"))


# --- Step 3: Create individual specialized agents ---

# 3.1 Web Search Agent:
# This agent is responsible for fetching real-time data from the internet.
# It uses DuckDuckGo as a search tool to find up-to-date info on any topic.
web_search = Agent(
    name="Web Search Agent",  # Friendly name for the agent
    role="search the web for relevant information to answer user queries accurately.",  # Defines the agent's task
    model=Groq(id="llama-3.3-70b-versatile"),  # Uses a powerful Llama 3.3 70B model hosted via Groq
    tools=[DuckDuckGo()],  # Adds web search capabilities via DuckDuckGo
    instructions=["Always include sources in your answers."],  # Ensures transparency with source links
    show_tool_calls=True,  # Logs tool usage for debugging or transparency
    markdown=True,         # Enables Markdown formatting for clean, readable responses
)


# 3.2 Financial Agent:
# This agent handles stock market and company-related financial data.
# It uses YFinanceTools to fetch details such as stock prices, news, and fundamentals.
financial_agent = Agent(
    name="Financial Agent",  # Name to identify this agent
    role="assist users with financial data analysis and stock market information.",  # Describes its role
    model=Groq(id="llama-3.3-70b-versatile"),  # Same Groq LLM for consistent quality
    tools=[
        YFinanceTools(
            company_news=True,               # Fetch latest company news
            stock_price=True,                # Get historical & current stock prices
            analyst_recommendations=True,    # Include analysts’ buy/sell/hold ratings
            stock_fundamentals=True          # Include company fundamentals (e.g., P/E ratio, revenue)
        )
    ],
    instructions=["use tables to display the data"],  # Ensures structured and easy-to-read output
    show_tool_calls=True,
    markdown=True,
)


# --- Step 4: Create a Multi-Agent Coordinator ---
# The Multi AI Agent acts as a "team leader" — it delegates tasks to specialized sub-agents.
# It can use outputs from both the web_search and financial_agent to generate comprehensive answers.
multi_ai_agent = Agent(
    name="Multi AI Agent",               # Main agent coordinating multiple sub-agents
    team=[web_search, financial_agent],  # The “team” of agents that work under it
    model=Groq(id="llama-3.3-70b-versatile"),  # Uses the same core Groq model
    instructions=[
        "use tables to display the data",            # Maintain consistency in output format
        "Always include sources in your answers."    # Reinforce transparency
    ],
    show_tool_calls=True,
    markdown=True,
)


# --- Step 5: Run the Multi-Agent query ---
# The multi_ai_agent will:
#   1. Ask the Financial Agent for Apple's stock performance and metrics.
#   2. Ask the Web Search Agent for recent news impacting Apple.
#   3. Combine both into one detailed report (in Markdown with tables and sources).
multi_ai_agent.print_response(
    "Provide a summary of Apple's stock performance over the past year, including recent news that may impact its future performance."
)
