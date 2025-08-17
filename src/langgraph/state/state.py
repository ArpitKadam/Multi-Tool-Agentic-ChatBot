from typing_extensions import TypedDict, Annotated
from typing import List, Dict, Any, Optional
from langgraph.graph.message import add_messages


class State(TypedDict):
    """
    Represents the structure of the state passed between graph nodes.

    Attributes:
        messages: A list of messages representing the conversation history.
                  This is managed by LangGraph to accumulate messages.
        frequency: The time frame for fetching news (e.g., 'daily', 'weekly').
        news_data: A list of raw news articles fetched from the search tool.
        summary: The final, formatted markdown summary of the news.
        filename: The path to the saved markdown file containing the summary.
    """
    messages: Annotated[List[Dict[str, Any]], add_messages]
    frequency: Optional[str]
    news_data: Optional[list]
    summary: Optional[str]
    filename: Optional[str]