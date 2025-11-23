import os
from typing import TypedDict, Annotated, Sequence, List, Any, Dict, Optional
from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs.chat_generation import ChatGeneration, ChatResult
from openai import OpenAI
import json
import csv
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Custom NVIDIA Chat Model
class NvidiaChatModel(BaseChatModel):
    """Custom LangChain wrapper for NVIDIA NIMS LLM"""
    
    client: Any = None
    model_name: str = "qwen/qwen3-next-80b-a3b-instruct"
    temperature: float = 0.6
    top_p: float = 0.7
    max_tokens: int = 4096
    api_key: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load API key from environment variable if not explicitly provided
        if not self.api_key:
            self.api_key = os.getenv("NVIDIA_API_KEY", "")
            if not self.api_key:
                raise ValueError(
                    "NVIDIA_API_KEY not found. Please set it as an environment variable or pass it to NvidiaChatModel."
                )

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response from NVIDIA NIMS API"""
        
        # Convert LangChain messages to NVIDIA API format
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})

        # Make the API call
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=formatted_messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=stop,
            stream=False
        )

        # Extract content from the response
        response_content = completion.choices[0].message.content
        
        # Create an AIMessage
        ai_message = AIMessage(content=response_content)
        
        # Create a ChatGeneration object
        chat_generation = ChatGeneration(message=ai_message, text=response_content)
        
        # Wrap it in a ChatResult
        return ChatResult(generations=[chat_generation])

    @property
    def _llm_type(self) -> str:
        return "nvidia_chat_model"


# Initialize LLM and Tavily
def initialize_llm_and_tools():
    """Initialize NVIDIA Chat Model and Tavily Search"""
    
    llm = NvidiaChatModel(
        model_name="qwen/qwen3-next-80b-a3b-instruct",
        temperature=0.6,
        top_p=0.7,
        max_tokens=4096,
    )
    
    tavily_tool = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
    
    return llm, tavily_tool


# Vector Store Management
def create_vector_store():
    """
    Create a vector store from PDF documents.
    
    This function loads PDF documents, splits the text into chunks, 
    and creates embeddings using Google Generative AI.
    
    Returns:
        FAISS: A vector store containing the text and corresponding embeddings.
    """
    
    try:
        # Try to load the existing vector store first
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-exp-03-07", 
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        print("Loaded existing vector store.")
        return vector_store
    except Exception as e:
        print(f"Could not load existing vector store: {e}. Creating new vector store...")
    
    # Loading documents
    documents = []
    try:
        for i in range(1, 4):
            try:
                loader = PyPDFLoader(f"document{i}.pdf")
                documents.extend(loader.load())
            except FileNotFoundError:
                print(f"File document{i}.pdf not found, skipping.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        # Fallback to loading just the first document
        try:
            loader = PyPDFLoader("document_1.pdf")
            documents.extend(loader.load())
        except FileNotFoundError:
            print("Critical: No documents found")
            return None
    
    # Splitting documents and creating text representations
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Number of text chunks: {len(texts)}")
    
    # Create embeddings using a Google Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-exp-03-07", 
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    # Create a vector store using FAISS from the provided text chunks and embeddings
    vector_store = FAISS.from_documents(texts, embedding=embeddings)
    
    # Save the vector store locally with the name "faiss_index"
    vector_store.save_local("faiss_index")
    return vector_store


# Tool Functions
def search_vector_store(query: str, vector_store) -> str:
    """Search the vector store for relevant information"""
    try:
        print(f"Starting vector search for query: {query}")
        results = vector_store.similarity_search(query, k=1)
        print(f"Vector search results: {results}")
        return results[0].page_content
    except Exception as e:
        print(f"Error searching vector store: {e}")
        return "No information found in the vector store."


def execute_tavily_tool(query: str, tavily_tool) -> str:
    """Execute Tavily web search"""
    try:
        print(f"Starting Tavily search for query: {query}")
        results = tavily_tool.run(query)
        print(f"Tavily search results: {results}")
        return results
    except Exception as e:
        print(f"Error in Tavily search: {e}")
        return "No information found using Tavily."


def create_tools(vector_store, tavily_tool):
    """Create tool instances for the agent"""
    
    # Create a tool for the vector store
    vector_search_tool = Tool(
        name="VectorSearch",
        func=lambda query: search_vector_store(query, vector_store),
        description="Searches the vector store for relevant information about diseases."
    )

    # Create a tool for Tavily search
    tavily_web_search_tool = Tool(
        name="TavilySearch",
        func=lambda query: execute_tavily_tool(query, tavily_tool),
        description="Searches the web for relevant information about diseases."
    )

    return [vector_search_tool, tavily_web_search_tool]


# Agent Setup
def create_agent(llm, tools):
    """Create the ReAct agent with tools"""
    
    prompt = PromptTemplate.from_template(
        """You are a medical information retrieval agent. Your task is to find information about diseases and their ICD codes.

    Tools you can use:
    {tools}

    Tool name: {tool_names}

    Human: {human_input}
    You must respond in the following format:
        Thought: <Your thought process>
        Action: <The action you will take>
        Action Input: <The input for the action>
        Observation: <The result of the action>
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I have gathered all the necessary information and will now provide the final answer.
        Final Answer: <The final answer in JSON format>

    Always use the following steps:
    1. **Retrieve Disease Description:** Use VectorSearch to retrieve the disease description.
    2. **Retrieve ICD Codes:**
        a. First, attempt to use VectorSearch to find the ICD codes for the disease.
        b. If VectorSearch does not provide the ICD codes, then use TavilySearch to retrieve them.
    3. Compile the results into the specified JSON format.
    4. Once you have compiled the JSON response, you MUST output it using the "Final Answer:" prefix.

    The final answer MUST be a JSON string with the following structure:
    {{
        "disease": "<disease_name>",
        "description": "<description>",
        "icd_codes": ["<code1>", "<code2>", ...]
    }}
    If VectorSearch does not provide a description, clearly state that in the "description" field of the final JSON.
    If VectorSearch does not provide ICD codes, clearly state that in the "icd_codes" field (e.g., ["Not found in vector store, retrieved from web search"]).

    {agent_scratchpad}
    """
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=7,
        verbose=True,
        return_intermediate_steps=True,
        max_execution_time=120.0
    )
    
    return agent_executor


# LangGraph State Management
class AgentState(TypedDict):
    """Type definition for agent state"""
    human_input: str
    messages: Sequence[HumanMessage]
    results: dict


def agent_node(state: AgentState, agent_executor):
    """Process the agent's response"""
    try:
        # Execute the agent
        result = agent_executor.invoke({
            "human_input": state["human_input"]
        })
        
        print("Raw result from agent:", result)
        
        # Check if the result contains the final JSON output
        if isinstance(result, dict) and "disease" in result and "description" in result and "icd_codes" in result:
            print("Final result obtained. Terminating chain.")
            return {
                "messages": state.get("messages", []),
                "results": result
            }
        
        # Continue processing if the result is incomplete
        return {
            "messages": state.get("messages", []),
            "results": result.get("output", "No output from agent")
        }
    except Exception as e:
        print(f"Error in agent_node: {e}")
        return {
            "messages": state.get("messages", []),
            "results": f"Error processing request: {str(e)}"
        }


def create_graph(agent_executor):
    """Create the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    # Add node with lambda to pass agent_executor
    workflow.add_node("agent", lambda state: agent_node(state, agent_executor))
    
    workflow.set_entry_point("agent")
    workflow.set_finish_point("agent")
    
    graph = workflow.compile()
    return graph


# Execution Functions
def run_graph(query: str, graph):
    """Run the agent graph with the given query"""
    inputs = {
        "human_input": query,
        "messages": [],
        "results": {}
    }
    try:
        result = graph.invoke(inputs)
        
        # Check if the result contains the final JSON output
        if isinstance(result, dict) and "disease" in result["results"] and "description" in result["results"] and "icd_codes" in result["results"]:
            print("Final result obtained. Stopping execution.")
            return result["results"]
        
        return result["results"]
    except Exception as e:
        print(f"Error running graph: {e}")
        return f"Error: {str(e)}"


def generate_output(query: str, result):
    """Convert the agent's output to JSON and CSV formats"""
    try:
        # Try to parse the result as JSON first
        try:
            output = json.loads(result) if isinstance(result, str) else result
        except (json.JSONDecodeError, TypeError):
            # If not a valid JSON, try to extract JSON from text
            try:
                start_idx = result.find('{')
                end_idx = result.rfind('}')
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = result[start_idx:end_idx+1]
                    output = json.loads(json_str)
                else:
                    output = {
                        "disease": query,
                        "description": result.split('\n')[0] if '\n' in result else result[:100],
                        "icd_codes": []
                    }
            except Exception:
                output = {
                    "disease": query,
                    "description": "Failed to parse agent output",
                    "icd_codes": []
                }
        
        # Extract relevant fields with proper error handling
        disease = output.get("disease", query)
        description = output.get("description", "No description available")
        icd_codes = output.get("icd_codes", [])
        
        if isinstance(icd_codes, str):
            icd_codes = [icd_codes]
        
        icd_codes_str = ", ".join(icd_codes) if icd_codes else "No ICD codes found"
        
        # Generate JSON output
        json_output = {
            "disease": disease,
            "description": description,
            "icd_codes": icd_codes
        }
        
        # Generate CSV output
        csv_output = [["Disease", "Description", "ICD Codes"]]
        csv_output.append([disease, description, icd_codes_str])
        
        return json_output, csv_output
    except Exception as e:
        print(f"Error generating output: {e}")
        json_output = {"error": str(e), "query": query}
        csv_output = [["Error", "Query"], [str(e), query]]
        return json_output, csv_output


# Main Function
def main(query: str):
    """Main function to run the agent and save results"""
    
    print(f"Processing query: {query}")
    
    # Initialize components
    llm, tavily_tool = initialize_llm_and_tools()
    vector_store = create_vector_store()
    
    if vector_store is None:
        raise ValueError("Failed to create vector store")
    
    tools = create_tools(vector_store, tavily_tool)
    agent_executor = create_agent(llm, tools)
    graph = create_graph(agent_executor)
    
    # Run the graph
    result = run_graph(query, graph)
    print(f"Result from graph: {result}")
    
    # Generate outputs
    json_output, csv_output = generate_output(query, result)
    
    # Save JSON output
    with open('output.json', 'w') as f:
        json.dump(json_output, f, indent=2)
    
    # Save CSV output
    with open('output.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_output)
    
    print("JSON output:", json_output)
    print("CSV output has been saved to output.csv")
    
    return json_output


if __name__ == "__main__":
    main("Gallstones")