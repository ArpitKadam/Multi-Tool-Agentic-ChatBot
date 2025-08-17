import json
import streamlit as st
import os
from typing import  Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class DisplayResultStreamlit:
    """
    Handles the rendering of graph results within the Streamlit UI.

    This class orchestrates the interaction with the LangGraph agent for different
    use cases and displays the output, including user messages, assistant responses,
    and tool execution details, in a chat-like interface.
    """

    def __init__(self, usecase: str, graph, user_message: str):
        """
        Initializes the result display handler.

        Args:
            usecase (str): The selected use case (e.g., "Basic ChatBot").
            graph (CompiledGraph): The compiled LangGraph agent to be executed.
            user_message (str): The initial message or prompt from the user.
        """
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message

    def _render_message(self, role: str, content: Any, is_tool: bool = False, tool_name: Optional[str] = None):
        """
        A generic utility to render a message in the Streamlit chat UI.

        Args:
            role (str): The role of the message sender (e.g., "user", "assistant").
            content (Any): The content of the message.
            is_tool (bool): Flag to indicate if this is a tool execution message.
            tool_name (Optional[str]): The name of the tool being executed.
        """
        with st.chat_message(role):
            if not is_tool:
                st.markdown(content)
                return

            # --- UI for Tool Execution ---
            st.markdown(f"**🔧 Using Tool: `{tool_name}`**")
            with st.expander("Click to see tool output", expanded=False):
                # Try to parse content as JSON for rich display
                try:
                    data = json.loads(content) if isinstance(content, str) else content
                    st.json(data) # Display the raw JSON for transparency

                    # Display images if available
                    if "images" in data and data["images"]:
                        st.markdown("---")
                        st.markdown("##### 📸 Images Found")
                        # Use columns for a cleaner layout
                        cols = st.columns(len(data["images"]))
                        for i, img_url in enumerate(data["images"]):
                            with cols[i]:
                                st.image(img_url, use_column_width=True)
                    
                    # Display web search results if available
                    if "results" in data and data["results"]:
                        st.markdown("---")
                        st.markdown("##### 🔗 Search Results")
                        for r in data["results"]:
                            title = r.get("title", "No title")
                            url = r.get("url", "#")
                            st.markdown(f"- [{title}]({url})")

                except (json.JSONDecodeError, TypeError):
                    # Fallback for non-JSON content
                    st.write(content)

    def display_result_on_ui(self):
        """
        Main entry point to execute the graph and render the entire conversation flow.
        """
        # Immediately display the user's message
        self._render_message("user", self.user_message)

        try:
            # --- Route to the appropriate handler based on the use case ---
            if self.usecase == "Basic ChatBot":
                self._handle_basic_chatbot()
            elif self.usecase == "ChatBot with Tools":
                self._handle_chatbot_with_tools()
            elif self.usecase == "AI News":
                self._handle_ai_news()
            else:
                st.warning(f"⚠️ Unknown use case: '{self.usecase}'. Please select a valid option.")

        except Exception as e:
            st.error(f"🚨 **An error occurred:**\n\nAn unexpected issue was encountered while processing your request. Please check your configuration and try again.\n\n*Details: {e}*")
            st.stop()

# In display_results.py, replace the existing _handle_basic_chatbot method with this one:

    def _handle_basic_chatbot(self):
        """Handles streaming response for the Basic ChatBot use case."""
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            with st.spinner("🤔 Thinking..."):
                # The input should be a list of messages for the graph state
                initial_input = {"messages": [HumanMessage(content=self.user_message)]}

                for event in self.graph.stream(initial_input):
                    for value in event.values():
                        # The 'messages' key holds a LIST of new messages from the node
                        if new_messages := value.get("messages"):
                            # Iterate through the list (usually contains one message)
                            for msg in new_messages:
                                if msg.content:
                                    full_response += msg.content
                                    # Update the placeholder with the accumulating response
                                    response_placeholder.markdown(full_response + "▌")
            
            # Display the final, complete response
            response_placeholder.markdown(full_response)

    def _handle_chatbot_with_tools(self):
        """Handles the response flow for the ChatBot with Tools use case."""
        with st.spinner("🔎 Thinking & using tools..."):
            # The graph is invoked, and it runs the full tool-use cycle
            initial_state = {"messages": [HumanMessage(content=self.user_message)]}
            response = self.graph.invoke(initial_state)

        messages = response.get("messages", [])
        
        # Keep track of the last AIMessage that contained tool calls
        last_ai_message_with_tool_calls = None

        for message in messages:
            if isinstance(message, HumanMessage):
                # This is the initial user message, which is already displayed.
                continue

            if isinstance(message, AIMessage):
                if message.tool_calls:
                    # This is the AI's decision to call a tool. We store it
                    # but don't display anything yet. The tool's result will be displayed.
                    last_ai_message_with_tool_calls = message
                if message.content:
                    # This is a final text response from the assistant after using a tool.
                    self._render_message("assistant", message.content)
            
            elif isinstance(message, ToolMessage):
                tool_name = "Unknown Tool"
                if last_ai_message_with_tool_calls:
                    matching_tool_call = next(
                        (tc for tc in last_ai_message_with_tool_calls.tool_calls if tc['id'] == message.tool_call_id),
                        None
                    )
                    if matching_tool_call:
                        tool_name = matching_tool_call['name']
                
                self._render_message("assistant", message.content, is_tool=True, tool_name=tool_name)

    def _handle_ai_news(self):
        """Handles the AI News fetching and summarization use case."""
        with st.spinner("📰 Fetching and summarizing the latest AI news..."):
            result = self.graph.invoke({"messages": [HumanMessage(content=self.user_message)]})

        summary = result.get("summary")
        if summary:
            with st.chat_message("assistant"):
                st.subheader("📰 AI News Summary")
                st.markdown(summary, unsafe_allow_html=True)
                # Provide a download button for the generated report
                if filename := result.get("filename"):
                    with open(filename, "r", encoding="utf-8") as f:
                        st.download_button(
                            label="📥 Download Full Report",
                            data=f.read(),
                            file_name=os.path.basename(filename),
                            mime="text/markdown",
                        )
        else:
            st.error(f"🚨 Could not generate a news summary for the selected timeframe.")