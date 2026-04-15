# Research Assistant Agent

**Cambrian College — Graduate Certificate in Artificial Intelligence**
**Natural Language Processing — Assignment 2**

> GitHub Repository: https://github.com/jemishdankhara008/nlp-assignment2


| Member | Contribution |
|---|---|
| Jemish Dankhara (A00316802) | Full implementation: LangGraph pipeline, MCP server, config, README |

---

---

## Project Description

This project implements **Option A: Research Assistant Agent** — an AI-powered pipeline that accepts a research topic from the user, automatically searches the web, extracts and summarizes the most relevant content using OpenAI GPT, and persists the structured research note to a local SQLite database through a custom MCP (Model Context Protocol) server.

The agent is orchestrated with **LangGraph** (graph-based multi-step workflow) and exposes its database layer through a **stdio-based MCP server** built with the official MCP Python SDK, ensuring clean decoupling between the agent logic and storage back-end.

---

## Architecture

```
User Input (query)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│                  LangGraph Pipeline (agent.py)            │
│                                                           │
│  ┌─────────────┐   ┌─────────────┐   ┌────────────────┐  │
│  │ search_node │──▶│ extract_node│──▶│ summarize_node │  │
│  │  (Tavily)   │   │  (cleanup)  │   │  (OpenAI GPT)  │  │
│  └─────────────┘   └─────────────┘   └────────┬───────┘  │
│                                               │           │
│                                       ┌───────▼────────┐  │
│                                       │   store_node   │  │
│                                       │  (MCP client)  │  │
│                                       └───────┬────────┘  │
└───────────────────────────────────────────────┼───────────┘
                                                │ stdio (MCP)
                                    ┌───────────▼───────────┐
                                    │    mcp_server.py       │
                                    │  ┌─────────────────┐  │
                                    │  │  save_research  │  │
                                    │  │  list_research  │  │
                                    │  │ search_research │  │
                                    │  └────────┬────────┘  │
                                    │           │            │
                                    │    ┌──────▼──────┐    │
                                    │    │  SQLite DB  │    │
                                    │    │ research.db │    │
                                    │    └─────────────┘    │
                                    └───────────────────────┘
```

### Node Descriptions

| Node | Responsibility |
|---|---|
| `search_node` | Calls Tavily Search API; returns top-N results with URLs and snippets |
| `extract_node` | Cleans results; concatenates text chunks; collects source URLs |
| `summarize_node` | Sends extracted content to OpenAI GPT with a structured summarization prompt |
| `store_node` | Connects to MCP server via stdio; calls `save_research` tool to persist the note |

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/jemishdankhara008/nlp-assignment2.git
cd nlp-assignment2
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

- Get an OpenAI key at: https://platform.openai.com/api-keys
- Get a free Tavily key (1 000 searches/month) at: https://tavily.com

---

## Usage

```bash
python agent.py "Explain retrieval augmented generation"
```

### Example Output

```
============================================================
Research Assistant Agent — Topic: Explain retrieval augmented generation
============================================================

[search_node] Searching for: Explain retrieval augmented generation
[search_node] Found 5 results.
[extract_node] Extracting content from search results...
[extract_node] Extracted 8214 characters from 5 sources.
[summarize_node] Summarizing with OpenAI (gpt-4o-mini)...
[summarize_node] Summary generated (874 chars).
[store_node] Saving research via MCP server...
[store_node] MCP tools available: ['save_research', 'list_research', 'search_research']
[store_node] MCP response: Research 'Explain retrieval augmented generation' saved successfully.

============================================================
RESEARCH SUMMARY
============================================================
Retrieval Augmented Generation (RAG) is a technique that enhances large language
models by grounding responses in retrieved, up-to-date external knowledge...

Key Points:
• RAG retrieves relevant documents from a vector store before generating a response.
• ...

Sources:
  - https://aws.amazon.com/what-is/retrieval-augmented-generation/
  - ...

MCP Status: Research 'Explain retrieval augmented generation' saved successfully.
============================================================
```

---

## MCP Server

The MCP server (`mcp_server.py`) runs as a separate subprocess and communicates over **stdio** using the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk). It abstracts all SQLite operations so the agent never touches the database directly.

### Exposed Tools

| Tool | Parameters | Description |
|---|---|---|
| `save_research` | `title`, `summary`, `sources` (list), `timestamp` (optional) | Insert a new research note into SQLite. |
| `list_research` | *(none)* | Return all saved entries (id, title, timestamp). |
| `search_research` | `keyword` | Full-text keyword search over title and summary fields. |

You can also run the MCP server standalone for testing:

```bash
python mcp_server.py
```

---

## Project Structure

```
nlp-assignment2/
├── agent.py          # LangGraph pipeline (4 nodes)
├── mcp_server.py     # MCP stdio server (SQLite tools)
├── config.py         # API keys + settings loader
├── requirements.txt  # Pinned dependencies
├── .env.example      # Environment variable template
├── .gitignore
└── README.md
```

