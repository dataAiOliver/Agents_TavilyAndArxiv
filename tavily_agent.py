import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import AzureChatOpenAI
from tavily import TavilyClient

# --- Load environment variables ---
load_dotenv()

# --- Tavily Tool: Multi-Result Web Search ---
def tavily_search(query: str, max_results: int = 20) -> str:
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(
        query=query,
        include_answer=True,
        include_raw_content=True,
        max_results=max_results,
    )
    results = response.get("results", [])
    if not results:
        return "No sources found."

    entries = []
    for result in results:
        entries.append(
            f"🔗 {result.get('title')} ({result.get('url')})\n"
            f"📝 {result.get('content', '')[:400].strip()}...\n"
        )

    return "\n\n".join(entries)

# --- LangChain Tool Definition ---
tavily_tool = Tool(
    name="TavilySearch",
    func=tavily_search,
    description=(
        "Use this tool to search the web for current information. "
        "It returns a list of summarized results with URLs and content snippets. "
        "You should read and extract key insights."
    )
)

# --- Azure OpenAI LLM Setup ---
llm = AzureChatOpenAI(
    temperature=0,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_SHORT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    model="gpt-4o",
)

# --- Agent Initialization ---
agent = initialize_agent(
    tools=[tavily_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# --- Task Prompt ---
query = (
    "Use the TavilySearch tool to find the most recent news on emerging AI startups in Europe "
    "as of July 2025. Then summarize key companies, their focus areas, and funding trends."
)

# --- Run Agent ---
response = agent.run(query)

# --- Output ---
print("\n--- Agent Response ---\n")
print(response)
