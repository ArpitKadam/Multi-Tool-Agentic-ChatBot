from typing_extensions import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

class State(TypedDict):
    """
    Represents the structure of the state used in graph
    """
    messages: Annotated[list, add_messages]