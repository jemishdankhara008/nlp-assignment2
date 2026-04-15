"""
config.py — Central configuration for the Research Assistant Agent.
Loads API keys and settings from environment variables via .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_openai_key() -> str:
    """Return the OpenAI API key from environment. Raises if missing."""
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Check your .env file.")
    return key


def get_tavily_key() -> str:
    """Return the Tavily Search API key from environment. Raises if missing."""
    key = os.getenv("TAVILY_API_KEY", "")
    if not key:
        raise EnvironmentError("TAVILY_API_KEY is not set. Check your .env file.")
    return key


# OpenAI model to use for summarization
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Max tokens for LLM summarization response
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))

# Number of Tavily search results to fetch
TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

# SQLite database file path
DB_PATH: str = os.getenv("DB_PATH", "research.db")

# MCP server script path
MCP_SERVER_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")
