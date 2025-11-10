from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os

# --- Load environment variables ---
load_dotenv()

# --- Print sanity check ---
print("✅ GROQ_API_KEY loaded:", bool(os.getenv("GROQ_API_KEY")))
print("✅ PHI_DEFAULT_MODEL:", os.getenv("PHI_DEFAULT_MODEL"))

# --- Force Groq use and disable OpenAI ---
os.environ.pop("OPENAI_API_KEY", None)
os.environ["PHI_DEFAULT_MODEL"] = "groq"

# --- Agents ---
web_search = Agent(
    name="Web Search Agent",
    role="search the web for relevant information to answer user queries accurately.",
    model = Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources in your answers."],
    show_tool_calls=True,
    markdown=True,
)

financial_agent = Agent(
    name="Financial Agent",
    role="assist users with financial data analysis and stock market information.",
    model = Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(
            company_news=True,
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True
        )
    ],
    instructions=["use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent = Agent(
    name="Multi AI Agent",
    team=[web_search, financial_agent],
    model = Groq(id="llama-3.3-70b-versatile"),
    instructions=[
        "use tables to display the data",
        "Always include sources in your answers."
    ],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response(
    "Provide a summary of Apple's stock performance over the past year, including recent news that may impact its future performance."
)
