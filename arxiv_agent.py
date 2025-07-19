from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import arxiv
import os

load_dotenv()

# --- Tool-Funktion: Nur Paper-Inhalte liefern, nicht interpretieren ---
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Returns abstracts of the latest papers on a topic from arXiv."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    entries = []
    for result in search.results():
        entries.append(
            f"📅 {result.published.date()}\n"
            f"📌 {result.title}\n"
            f"👥 Authors: {', '.join(str(a) for a in result.authors)}\n"
            f"📝 Abstract: {result.summary.strip()}\n"
        )

    return "\n\n".join(entries) if entries else "No results found."

# --- Tool definieren ---
arxiv_tool = Tool(
    name="ArxivSearch",
    func=arxiv_search,
    description=(
        "Use this tool to find abstracts of the latest scientific papers on a topic from arXiv. "
        "The result is a list of paper titles, authors, and abstracts. You should read and summarize them."
    )
)

# --- LLM initialisieren ---
llm = AzureChatOpenAI(
    temperature=0,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_SHORT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    model="gpt-4o",
)

# --- Agent initialisieren ---
agent = initialize_agent(
    tools=[arxiv_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# --- Agent-Query ---
query = (
    "Use the ArxivSearch tool to get recent papers on multimodal large language models, "
    "then provide a detailed and structured summary of what the current research focuses on, including key techniques and trends."
)

response = agent.run(query)

print("\n--- Agent Response ---\n")
print(response)
