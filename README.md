# langgraph-ICD

Lightweight prototype that uses LangGraph + LangChain tools to retrieve disease descriptions and ICD codes using a local FAISS vector store and web search (Tavily). Main entrypoint is the Jupyter notebook `icd_agent.ipynb`.

## Features
- Build a FAISS vector store from PDF documents.
- Search for disease descriptions via embeddings (Google Generative AI embeddings).
- Fall back to Tavily web search for ICD codes if not present in the vector store.
- Agent workflow implemented with LangChain React-style agent and LangGraph StateGraph.
- Outputs final structured result to `output.json` and `output.csv`.

## Quickstart (Windows)
1. Clone / open project folder in VS Code.
2. Install dependencies (run in terminal / VS Code integrated terminal):
   ```
   pip install langgraph langchain langchain-google-genai langchain-community faiss-cpu openai pypdf python-dotenv
   ```
3. Place your PDFs as `document1.pdf`, `document2.pdf`, ... in the project root (or update the loader paths in the notebook).
4. Create a `.env` file in the project root or set environment variables in your shell:
   - NVIDIA_API_KEY — (optional) for NVIDIA chat integration
   - GEMINI_API_KEY — required for Google Generative embeddings
   - TAVILY_API_KEY — required for Tavily search
   Example `.env` (used by python-dotenv):
   ```
   NVIDIA_API_KEY=your_nvidia_key
   GEMINI_API_KEY=your_gemini_key
   TAVILY_API_KEY=your_tavily_key
   ```
   On Windows PowerShell you can temporarily set an env var:
   ```
   $env:GEMINI_API_KEY="your_gemini_key"
   ```

5. Open and run `icd_agent.ipynb` in VS Code / Jupyter. The notebook contains cells to:
   - Install packages
   - Define NvidiaChatModel wrapper
   - Create / load FAISS index from PDFs
   - Build & run the agent
   - Save `output.json` and `output.csv`

6. Example run from notebook:
   - The notebook's `main("Gallstones")` will execute the StateGraph agent and write `output.json` / `output.csv`.

## Files of interest
- `icd_agent.ipynb` — primary notebook implemented using NVIDIA NIMS LLM (custom NvidiaChatModel wrapper / NVIDIA chat).
- `icd_agent_gemini.ipynb` — secondary notebook implemented using Google Gemini LLM (ChatGoogleGenerativeAI).
- `output.json`, `output.csv` — produced by `main()` after running the agent.
- PDF documents (document1.pdf, ...)

## Notes & troubleshooting
- Ensure the GEMINI_API_KEY is valid and billing enabled for embeddings.
- FAISS index will be saved locally as `faiss_index` by `create_vector_store()`.
- If PDFs are missing the loader will skip and the graph may return fallbacks.
- The Nvidia/OpenAI client usage in the notebook is a custom wrapper — confirm the client class and base_url match your SDK version.
- Increase `max_iterations` / `max_execution_time` in the agent if timeouts occur.

## Extending
- Swap embeddings provider or change chunk size in `CharacterTextSplitter`.
- Add more documents or a different vector store backend.
- Improve parsing in `generate_output()` for varied agent text formats.

No license specified — treat as experimental prototype.