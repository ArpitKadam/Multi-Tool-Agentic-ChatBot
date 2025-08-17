from typing_extensions import TypedDict, Annotated
from typing import List, Dict, Any
from langgraph.graph.message import add_messages


class State(TypedDict):
    """
    Represents the structure of the conversational state passed between graph nodes.

    Attributes:
        messages (List[Dict[str, Any]]): A list of messages representing the conversation
                                         history. Each message should follow the format
                                         expected by the LangGraph message schema.
    """
    messages: Annotated[List[Dict[str, Any]], add_messages]
