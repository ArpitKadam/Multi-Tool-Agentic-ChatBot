import os
from dotenv import load_dotenv

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch

from src.langgraph.state.state import State

# Load environment variables from a .env file
load_dotenv()
TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")


class AINewsNode:
    """A collection of nodes for fetching, summarizing, and saving AI news."""

    _OUTPUT_DIR = "./AINews"

    def __init__(self, llm: BaseLanguageModel):
        """
        Initializes the AINewsNode with a language model and the Tavily search client.

        Args:
            llm (BaseLanguageModel): An instance of a LangChain compatible language model.
        """
        self.llm = llm
        self.tavily = TavilySearch(
            api_key=TAVILY_API_KEY,
            max_results=10,
            search_depth="advanced",
            include_images=True,
            topic="news",
            verbose=True,
        )

    def fetch_news(self, state: State) -> State:
        """
        Fetches AI and technology news based on a frequency specified in the state.

        Args:
            state (State): The current graph state, expected to contain the frequency
                           in the last message.

        Returns:
            State: The updated state containing the fetched 'news_data' and 'frequency'.

        Raises:
            ValueError: If the frequency is missing or invalid, or if the fetch fails.
        """
        if not self.llm:
            raise ValueError("Language model not provided to AINewsNode.")

        try:
            if not state.get("messages"):
                raise ValueError("No frequency found in state messages.")
            
            # Extract frequency from the last message in the conversation
            frequency = state["messages"][-1].content.lower().strip()

            time_range_map = {"daily": "d", "weekly": "w", "monthly": "m", "yearly": "y"}
            if frequency not in time_range_map:
                valid_options = ", ".join(time_range_map.keys())
                raise ValueError(f"Invalid frequency: '{frequency}'. Must be one of: {valid_options}")

            response = self.tavily.invoke(
                input="Top 10 latest AI and technology-related news in India and globally.",
                time_range=time_range_map[frequency],
            )

            state["news_data"] = response
            state["frequency"] = frequency
            return state

        except Exception as e:
            # Wrap the original exception for better debugging
            raise ValueError(f"Failed to fetch AI News: {e}") from e

    def summarize_news(self, state: State) -> State:
        """
        Summarizes the fetched news articles into a reader-friendly markdown report.

        Args:
            state (State): The current graph state, expected to contain 'news_data'.

        Returns:
            State: The updated state with the generated 'summary'.

        Raises:
            ValueError: If 'news_data' is not found in the state.
        """
        news_items = state.get("news_data", {}).get("results", [])
        if not news_items:
            raise ValueError("No news data found in state. Please run fetch_news first.")

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an expert assistant that summarizes technology and AI news into a concise, reader-friendly markdown report.

                    **Instructions:**
                    - Summarize each article in **3-4 clear sentences**.
                    - Include the publication **date** in `YYYY-MM-DD` format.
                    - Sort the news items with the **latest appearing first**.
                    - Display Images using urls, using markdown syntax.
                    - Format each item exactly as follows:
                    ## [Article Title](URL)
                    #### Date: YYYY-MM-DD
                    #### Images: ![Image Alt Text](URL)
                    #### Summary: ...
                    """,
                ),
                ("human", "Please summarize the following articles:\n\n{articles}"),
            ]
        )

        articles_str = "\n---\n".join(
            [
                f"Title: {article.get('title', 'N/A')}\n"
                f"URL: {article.get('url', '#')}\n"
                f"Date: {article.get('published_date', 'N/A')}\n"
                f"Content: {article.get('content', 'No content available.')}"
                for article in news_items
            ]
        )

        chain = prompt_template | self.llm
        response = chain.invoke({"articles": articles_str})

        state["summary"] = response.content
        return state

    def save_result(self, state: State) -> State:
        """
        Saves the news summary to a markdown file.

        Args:
            state (State): The current graph state, expected to contain 'summary' and 'frequency'.

        Returns:
            State: The updated state with the 'filename' of the saved report.

        Raises:
            ValueError: If 'summary' or 'frequency' is not found in the state.
        """
        frequency = state.get("frequency")
        summary = state.get("summary")

        if not summary or not frequency:
            raise ValueError("Summary or frequency not found in state. Please run summarize_news first.")

        # Ensure the output directory exists
        os.makedirs(self._OUTPUT_DIR, exist_ok=True)
        
        filename = os.path.join(self._OUTPUT_DIR, f"{frequency}_summary.md")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# {frequency.capitalize()} AI News Summary\n\n")
            f.write(summary)
            
        print(f"✅ News summary saved to: {filename}")
        
        state["filename"] = filename
        return state