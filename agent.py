"""
agent.py — LangGraph-based Research Assistant Agent (OpenAI edition).

Workflow (4 nodes):
  search_node    → Calls Tavily Search API to fetch top results for the query.
  extract_node   → Cleans and extracts relevant text content from search results.
  summarize_node → Sends extracted content to OpenAI GPT for summarization.
  store_node     → Persists the summary to SQLite via the MCP server.

Usage:
    python agent.py "Explain retrieval augmented generation"
"""

import sys
import json
import textwrap
from typing import TypedDict

from openai import OpenAI
from tavily import TavilyClient
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, END

import config


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    """Shared state that flows through every node in the LangGraph pipeline."""
    query: str                 # The user's research topic
    search_results: list[dict] # Raw results from Tavily
    extracted_content: str     # Cleaned, concatenated text from search results
    summary: str               # LLM-generated summary
    sources: list[str]         # Source URLs extracted from search results
    status: str                # Final status message (from MCP store step)


# ---------------------------------------------------------------------------
# Node 1: search_node
# ---------------------------------------------------------------------------

def search_node(state: ResearchState) -> ResearchState:
    """
    Fetch top web results for the user query using the Tavily Search API.

    Updates:
        search_results — list of result dicts (title, url, content, score).
    """
    print(f"[search_node] Searching for: {state['query']}")
    try:
        client = TavilyClient(api_key=config.get_tavily_key())
        response = client.search(
            query=state["query"],
            max_results=config.TAVILY_MAX_RESULTS,
            include_raw_content=False,
        )
        results: list[dict] = response.get("results", [])
        print(f"[search_node] Found {len(results)} results.")
        return {**state, "search_results": results}
    except Exception as exc:
        print(f"[search_node] ERROR: {exc}", file=sys.stderr)
        return {**state, "search_results": [], "status": f"Search failed: {exc}"}


# ---------------------------------------------------------------------------
# Node 2: extract_node
# ---------------------------------------------------------------------------

def extract_node(state: ResearchState) -> ResearchState:
    """
    Extract and clean text content from raw Tavily search results.

    Concatenates snippets/content fields from each result and collects URLs.

    Updates:
        extracted_content — plain-text block ready for summarization.
        sources           — list of unique source URLs.
    """
    print("[extract_node] Extracting content from search results...")
    results: list[dict] = state.get("search_results", [])

    if not results:
        return {**state, "extracted_content": "", "sources": []}

    chunks: list[str] = []
    sources: list[str] = []

    for r in results:
        url: str = r.get("url", "")
        title: str = r.get("title", "")
        content: str = r.get("content", "") or r.get("snippet", "")

        if url:
            sources.append(url)
        if content:
            chunks.append(f"Source: {title}\nURL: {url}\n\n{content}")

    extracted: str = "\n\n---\n\n".join(chunks)
    print(f"[extract_node] Extracted {len(extracted)} characters from {len(sources)} sources.")
    return {**state, "extracted_content": extracted, "sources": sources}


# ---------------------------------------------------------------------------
# Node 3: summarize_node
# ---------------------------------------------------------------------------

def summarize_node(state: ResearchState) -> ResearchState:
    """
    Send the extracted content to OpenAI GPT for summarization.

    Produces a structured summary with key points and source references.

    Updates:
        summary — LLM-generated structured summary string.
    """
    print(f"[summarize_node] Summarizing with OpenAI ({config.OPENAI_MODEL})...")
    extracted: str = state.get("extracted_content", "")

    if not extracted:
        return {**state, "summary": "No content available to summarize."}

    system_prompt = (
        "You are a research assistant. Your job is to produce concise, "
        "well-structured research summaries from web source excerpts."
    )

    user_prompt = textwrap.dedent(f"""
        Below are excerpts from multiple web sources about the topic:

        TOPIC: {state['query']}

        SOURCES:
        {extracted}

        Please provide a structured research summary that includes:
        1. A brief overview (2–3 sentences)
        2. Key points (bullet list, at least 4 points)
        3. Important caveats or limitations

        Be factual, clear, and cite which sources support each key point where appropriate.
    """).strip()

    try:
        client = OpenAI(api_key=config.get_openai_key())
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            max_tokens=config.MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        summary: str = response.choices[0].message.content or ""
        print(f"[summarize_node] Summary generated ({len(summary)} chars).")
        return {**state, "summary": summary}
    except Exception as exc:
        print(f"[summarize_node] ERROR: {exc}", file=sys.stderr)
        return {**state, "summary": f"Summarization failed: {exc}"}


# ---------------------------------------------------------------------------
# Node 4: store_node
# ---------------------------------------------------------------------------

async def store_node(state: ResearchState) -> ResearchState:
    """
    Persist the research summary to SQLite via the MCP server.

    Uses langchain-mcp-adapters to connect to the stdio-based MCP server,
    discovers available tools, then calls save_research.

    Updates:
        status — confirmation message from the MCP server.
    """
    print("[store_node] Saving research via MCP server...")

    summary: str = state.get("summary", "")
    sources: list[str] = state.get("sources", [])
    query: str = state.get("query", "Untitled Research")

    if not summary:
        return {**state, "status": "Nothing to save — summary is empty."}

    try:
        mcp_client = MultiServerMCPClient(
            {
                "research-db": {
                    "command": "python",
                    "args": [config.MCP_SERVER_PATH],
                    "transport": "stdio",
                }
            }
        )
        tools = await mcp_client.get_tools()
        tool_names = [t.name for t in tools]
        print(f"[store_node] MCP tools available: {tool_names}")

        save_tool = next((t for t in tools if t.name == "save_research"), None)
        if save_tool is None:
            return {**state, "status": "ERROR: save_research tool not found on MCP server."}

        result = await save_tool.ainvoke({
            "title": query,
            "summary": summary,
            "sources": sources,
        })

        # Result may be a list of content blocks: [{'type': 'text', 'text': '...'}]
        if isinstance(result, list):
            text_blocks = [b.get("text", "") for b in result if isinstance(b, dict) and b.get("type") == "text"]
            result = " ".join(text_blocks)

        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                pass

        status_msg: str = (
            result.get("message", str(result))
            if isinstance(result, dict)
            else str(result)
        )
        print(f"[store_node] MCP response: {status_msg}")
        return {**state, "status": status_msg}

    except Exception as exc:
        print(f"[store_node] ERROR: {exc}", file=sys.stderr)
        return {**state, "status": f"MCP store failed: {exc}"}


# ---------------------------------------------------------------------------
# Build the LangGraph workflow
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Construct and compile the 4-node LangGraph research pipeline.

    Returns:
        A compiled LangGraph app ready to invoke.
    """
    workflow = StateGraph(ResearchState)

    workflow.add_node("search_node",    search_node)
    workflow.add_node("extract_node",   extract_node)
    workflow.add_node("summarize_node", summarize_node)
    workflow.add_node("store_node",     store_node)

    workflow.set_entry_point("search_node")
    workflow.add_edge("search_node",    "extract_node")
    workflow.add_edge("extract_node",   "summarize_node")
    workflow.add_edge("summarize_node", "store_node")
    workflow.add_edge("store_node",     END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_agent(query: str) -> None:
    """
    Execute the research agent for a given query and print the results.

    Args:
        query: The research topic to investigate.
    """
    print("\n" + "=" * 60)
    print(f"Research Assistant Agent — Topic: {query}")
    print("=" * 60 + "\n")

    app = build_graph()

    initial_state: ResearchState = {
        "query":             query,
        "search_results":    [],
        "extracted_content": "",
        "summary":           "",
        "sources":           [],
        "status":            "",
    }

    final_state: ResearchState = await app.ainvoke(initial_state)

    print("\n" + "=" * 60)
    print("RESEARCH SUMMARY")
    print("=" * 60)
    print(final_state["summary"])
    print("\nSources:")
    for url in final_state["sources"]:
        print(f"  - {url}")
    print(f"\nMCP Status: {final_state['status']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import asyncio

    if len(sys.argv) < 2:
        print("Usage: python agent.py \"<research topic>\"")
        sys.exit(1)

    topic = " ".join(sys.argv[1:])
    asyncio.run(run_agent(topic))
