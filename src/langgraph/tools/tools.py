import os
from typing import List
from dotenv import load_dotenv

# LangChain community tools & wrappers
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
    BraveSearch
)
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain_community.tools.google_jobs import GoogleJobsQueryRun
from langchain_tavily import TavilySearch
from langchain_community.utilities import (
    ArxivAPIWrapper,
    WikipediaAPIWrapper,
    GoogleFinanceAPIWrapper,
    DuckDuckGoSearchAPIWrapper,
    GoogleScholarAPIWrapper,
    SerpAPIWrapper,
    GoogleJobsAPIWrapper,
)

# LangGraph
from langgraph.prebuilt import ToolNode
from langchain.tools import Tool

# -----------------------------------------------------------------------------
# Load environment variables
# -----------------------------------------------------------------------------
load_dotenv()

SERP_API_KEY: str | None = os.getenv("SERP_API_KEY")
TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")
BRAVE_SEARCH_API_KEY: str | None = os.getenv("BRAVE_SEARCH_API_KEY")

if not SERP_API_KEY:
    raise ValueError("Missing required environment variable: SERP_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("Missing required environment variable: TAVILY_API_KEY")
if not BRAVE_SEARCH_API_KEY:
    raise ValueError("Missing required environment variable: BRAVE_SEARCH_API_KEY")

# -----------------------------------------------------------------------------
# API Wrappers
# -----------------------------------------------------------------------------
arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=5, load_max_docs=5, doc_content_chars_max=50000)
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=5, lang="en", doc_content_chars_max=5000)
duck_api_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
google_scholar_api_wrapper = GoogleScholarAPIWrapper(
    serp_api_key=SERP_API_KEY, top_k_results=5, hl="en"
)
google_finance_api_wrapper = GoogleFinanceAPIWrapper(
    serp_api_key=SERP_API_KEY, serp_search_engine="google_finance"
)
google_jobs_api_wrapper = GoogleJobsAPIWrapper(
    serp_api_key=SERP_API_KEY, serp_search_engine="google_jobs"
)
serp_hotels_api_wrapper = SerpAPIWrapper(
    search_engine="google_hotels", serpapi_api_key=SERP_API_KEY
)

# -----------------------------------------------------------------------------
# Tool Wrappers
# -----------------------------------------------------------------------------
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api_wrapper, verbose=True)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper, verbose=True)
duck_tool = DuckDuckGoSearchRun(api_wrapper=duck_api_wrapper, verbose=True)
tavily_tool = TavilySearch(api_key=TAVILY_API_KEY, verbose=True, max_results=5, include_images=True)
brave_tool = BraveSearch(search_kwargs={"max_results": 5}, verbose=True)
google_scholar_tool = GoogleScholarQueryRun(api_wrapper=google_scholar_api_wrapper, verbose=True)
google_finance_tool = GoogleFinanceQueryRun(api_wrapper=google_finance_api_wrapper, verbose=True)
google_jobs_tool = GoogleJobsQueryRun(api_wrapper=google_jobs_api_wrapper, verbose=True)

# Custom tool using SerpAPI for hotels
serp_hotel_tool = Tool(
    name="serp-search",
    description="A wrapper around Google Hotels using Serp API. Useful for hotel-related search queries.",
    func=serp_hotels_api_wrapper.run,
)

# -----------------------------------------------------------------------------
# Public Functions
# -----------------------------------------------------------------------------
def get_tools() -> List[Tool]:
    """
    Collects and returns the list of available tools for the chatbot.
    
    Returns:
        List[Tool]: A list of initialized LangChain tools.
    """
    return [
        arxiv_tool,
        wiki_tool,
        duck_tool,
        tavily_tool,
        brave_tool,
        google_scholar_tool,
        google_finance_tool,
        google_jobs_tool,
        serp_hotel_tool,
    ]


def create_tools_node(tools: List[Tool]) -> ToolNode:
    """
    Creates a LangGraph ToolNode from a given list of tools.
    
    Args:
        tools (List[Tool]): List of initialized tools.
    
    Returns:
        ToolNode: A LangGraph ToolNode that can be added to the chatbot graph.
    """
    return ToolNode(tools=tools)
