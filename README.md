# LangGraph ICD Agent

An intelligent implementation leveraging **LangGraph** and **LangChain** to automate disease information retrieval and ICD code lookup through a hybrid search architecture combining local vector embeddings and web intelligence.

## 📋 Overview

This system implements an intelligent agent workflow that:
- **Retrieves** disease descriptions from a local FAISS vector store (built from PDF knowledge bases)
- **Augments** search results via Tavily web search when vector store data is insufficient
- **Normalizes** outputs into structured JSON/CSV formats suitable for downstream processing
- **Supports** multiple LLM backends (NVIDIA NIMS, Google Gemini) for flexible deployment

### Architecture
```
User Query
    ↓
LangChain ReAct Agent (with LLM)
    ├── VectorSearch Tool (FAISS + Google Embeddings)
    └── TavilySearch Tool (Web API)
    ↓
LangGraph StateGraph Orchestration
    ↓
Structured Output (JSON/CSV)
```

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Dual LLM Support** | NVIDIA NIMS (primary) & Google Gemini (secondary) implementations |
| **Vector Store** | FAISS-backed local embeddings for low-latency retrieval |
| **Hybrid Search** | Vector + web search fallback for comprehensive coverage |
| **Agent Framework** | ReAct-style agent with tool composition via LangChain |
| **State Management** | Type-safe graph orchestration with LangGraph |
| **Multi-format Output** | JSON & CSV export for integrations |
| **PDF Ingestion** | Automatic document loading and chunking from `data/` folder |
| **Multiple Run Modes** | Jupyter Notebooks or standalone Python scripts |

## 🚀 Quickstart

### Prerequisites
- **Python** 3.9+
- **Windows/Linux/macOS**
- API Keys: GEMINI_API_KEY, TAVILY_API_KEY, (optionally) NVIDIA_API_KEY

### Installation

1. **Clone repository:**
   ```bash
   git clone <repo-url>
   cd langgraph-ICD
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # Windows PowerShell
   # or
   source venv/bin/activate     # macOS/Linux
   ```

3. **Install dependencies:**

   **For Jupyter Notebooks (One-go installation):**
   ```bash
   %pip install langgraph langchain langchain-google-genai langchain-community faiss-cpu openai pypdf python-dotenv
   ```
   *Run this command directly in the first notebook cell. Dependencies install in a single batch.*

   **For Python Script (.py file):**
   ```bash
   pip install -r requirements.txt
   ```
   *Or manually install all dependencies:*
   ```bash
   pip install langgraph langchain langchain-google-genai langchain-community faiss-cpu openai pypdf python-dotenv
   ```

4. **Configure environment:**
   Create `.env` file in project root:
   ```env
   # Required
   GEMINI_API_KEY=sk-...your-gemini-key...
   TAVILY_API_KEY=tvly-...your-tavily-key...

   # Optional (for NVIDIA backend)
   NVIDIA_API_KEY=nvapi-...your-nvidia-key...
   ```

   **Note:** Never commit `.env` to version control. Add to `.gitignore`:
   ```
   .env
   .env.local
   ```

5. **Prepare knowledge base:**
   Create a `data/` folder and place PDF documents inside:
   ```bash
   mkdir data
   # Copy PDFs into data folder
   cp *.pdf data/
   ```
   
   **Or manually:**
   ```
   data/
   ├── document1.pdf
   ├── document2.pdf
   └── document3.pdf
   ```

6. **Run the application:**

   **Option A: Jupyter Notebook**
   ```bash
   # VS Code
   code .
   # Then open icd_agent.ipynb or icd_agent_gemini.ipynb and run cells

   # Or Jupyter CLI
   jupyter notebook icd_agent.ipynb
   ```

   **Option B: Python Script**
   ```bash
   python icd_agent.py
   ```

## 📁 Project Structure

```
langgraph-ICD/
├── icd_agent.py                 # ⭐ PRIMARY: Standalone Python script (NVIDIA NIMS)
├── icd_agent.ipynb              # PRIMARY: Jupyter notebook (NVIDIA NIMS)
├── icd_agent_gemini.ipynb       # SECONDARY: Jupyter notebook (Google Gemini)
├── graph_visualizer.py          # Graph visualization utility
├── data/                        # Knowledge base PDFs (user-provided)
│   ├── document1.pdf
│   ├── document2.pdf
│   └── document3.pdf
├── faiss_index/                 # Auto-generated vector store (gitignored)
├── output.json                  # Query results (auto-generated)
├── output.csv                   # Query results (auto-generated)
├── .env                         # API keys (gitignored)
├── .gitattributes               # EOL normalization rules
├── requirements.txt             # Dependency manifest
└── README.md                    # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `GEMINI_API_KEY` | ✓ | Google Generative AI embeddings & LLM |
| `TAVILY_API_KEY` | ✓ | Web search fallback |
| `NVIDIA_API_KEY` | ✓ | NVIDIA NIMS LLM |

### Runtime Options

Edit cells in notebook or code in `icd_agent.py` to customize:

**Vector Store Chunk Size:**
```python
text_splitter = CharacterTextSplitter(
    chunk_size=1000,      # Increase for larger contexts
    chunk_overlap=100     # Overlap for boundary relevance
)
```

**Agent Iterations:**
```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=7,           # Tool invocation limit
    max_execution_time=120.0,   # Timeout in seconds
    verbose=True
)
```

**LLM Temperature (creativity vs consistency):**
```python
llm = NvidiaChatModel(
    temperature=0.6,    # 0.0 = deterministic, 1.0 = creative
    top_p=0.7,
    max_tokens=4096
)
```

## 📊 Usage Examples

### Basic Query via Python Script
```python
from icd_agent import main

result = main("fever")
# Output:
# {
#   "disease": "fever",
#   "description": "...",
#   "icd_codes": ["R50.9", "R50.0", ...]
# }
```

### Basic Query via Jupyter Notebook
```python
# Last cell in notebook:
main("Gallstones")  # Returns structured JSON output
```

### Batch Processing
```python
queries = ["diabetes", "hypertension", "pneumonia"]
results = [main(q) for q in queries]

# Export to CSV
import csv
with open('batch_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['disease', 'description', 'icd_codes'])
    writer.writerows(results)
```

## 🏗️ Architecture Details

### Agent Workflow (ReAct Pattern)

1. **Thought** → Agent analyzes query and selects tool
2. **Action** → Invoke VectorSearch or TavilySearch
3. **Observation** → Parse tool results
4. **Thought** → Decide: sufficient data? → YES: compile answer / NO: repeat
5. **Final Answer** → Structured JSON output

### Vector Store Pipeline

```
PDFs (from data/) → PyPDFLoader → Split (1000-token chunks) 
  → Google Embeddings → FAISS Index → Similarity Search
```

### State Management

```python
class AgentState(TypedDict):
    human_input: str           # User query
    messages: Sequence[HumanMessage]  # Conversation history
    results: dict              # Accumulated results
```

## ⚙️ Implementation Variants

| File | Type | LLM Provider | Strengths | Best For |
|------|------|-------------|-----------|----------|
| `icd_agent.py` | Script | NVIDIA NIMS | Cost-effective, low latency | Production, CLI usage, automation |
| `icd_agent.ipynb` | Notebook | NVIDIA NIMS | Interactive debugging, visualization | Development, testing |
| `icd_agent_gemini.ipynb` | Notebook | Google Gemini | Advanced reasoning, multimodal | Complex queries, research |

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `GEMINI_API_KEY not found` | Verify `.env` exists in project root; reload notebook kernel or restart Python script |
| `No PDF files found in 'data' directory` | Create `data/` folder and place PDFs inside: `mkdir data && cp *.pdf data/` |
| `FAISS index corrupted` | Delete `faiss_index/` folder; restart to regenerate from `data/` folder PDFs |
| Agent timeout after 120s | Increase `max_execution_time` or reduce knowledge base size |
| PDF line ending warnings | Run `git add --renormalize .` after adding `.gitattributes` |
| Low relevance results | Adjust `chunk_size` in `CharacterTextSplitter`; add more PDFs to `data/` folder |
| `ModuleNotFoundError` in .py script | Ensure virtual environment activated & `requirements.txt` installed: `pip install -r requirements.txt` |
| Notebook cells fail silently | Check `.env` file exists; verify API keys are valid; check internet connection for web search |

## 📈 Performance Tuning

**For Speed:**
- Reduce `chunk_size` (trade-off: less context per result)
- Lower `k` parameter in `similarity_search(query, k=1)` (fewer results)

**For Accuracy:**
- Increase `chunk_size` & `chunk_overlap`
- Raise `max_iterations` in agent
- Use `temperature=0.0` for deterministic outputs

**For Cost:**
- Cache embeddings: check FAISS save/load logic
- Batch queries to reuse single agent initialization
- Use NVIDIA backend instead of Gemini for high volume

## 🔐 Security Best Practices

- ✅ Store API keys in `.env` (add to `.gitignore`)
- ✅ Never log raw API keys; use redaction
- ✅ Validate user queries for injection attacks
- ✅ Implement rate limiting for web API calls
- ✅ Use `.gitattributes` to prevent binary corruption in PDFs
- ✅ Rotate API keys periodically
- ✅ Keep `data/` folder private; don't commit sensitive PDFs

## 📝 Output Formats

### JSON Structure
```json
{
  "disease": "Gallstones",
  "description": "Hardened deposits of digestive fluid...",
  "icd_codes": ["K80.0", "K80.1", "K80.2"]
}
```

### CSV Structure
```
Disease,Description,ICD Codes
Gallstones,Hardened deposits...,K80.0; K80.1; K80.2
```

## 📚 References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain ReAct Agent](https://python.langchain.com/docs/modules/agents/)
- [FAISS Index](https://github.com/facebookresearch/faiss)
- [ICD-10 Official](https://www.who.int/standards/classifications/classification-of-diseases)
- [Tavily Search API](https://tavily.com/)

## 📄 License

Experimental prototype. No license specified.

## 👥 Authors

- Annu (Owner)

---
