import json
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

class DisplayResultStreamlit:
    def __init__(self, usecase, graph, user_message):
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message

    def _display_message(self, role: str, content, expand_tool: bool = False):
        """
        Utility function to render messages in Streamlit chat UI.
        Handles both plain text and dict/JSON with images + results.
        """
        with st.chat_message(role):
            if expand_tool:
                st.markdown("🔧 **Tool Execution Started**")
                with st.expander("Tool Details", expanded=False):
                    st.write(content)

                    # ✅ Handle dict or JSON string
                data = None
                if isinstance(content, dict):
                    data = content
                else:
                    try:
                        data = json.loads(content)
                    except Exception:
                        pass

                if data:
                    # Show images
                    if "images" in data:
                        st.subheader("📸 Images")
                        for img_url in data["images"]:
                            st.image(img_url, use_container_width=True)

                    # Show results (links + titles)
                    if "results" in data:
                        st.subheader("🔗 Results")
                        for r in data["results"]:
                            title = r.get("title", "No title")
                            url = r.get("url", "#")
                            st.markdown(f"- [{title}]({url})")

                st.markdown("✅ **Tool Execution Finished**")
            else:
                st.write(content)


    def display_result_on_ui(self):
        """
        Main function to process chatbot responses and display them in UI
        """
        usecase = self.usecase
        graph = self.graph
        user_message = self.user_message

        # Show user input in UI immediately
        self._display_message("user", user_message)

        try:
            if usecase == "Basic ChatBot":
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Thinking..."):
                        for event in graph.stream({"messages": ("user", user_message)}):
                            for value in event.values():
                                if value.get("messages"):
                                    st.markdown(value["messages"].content)

            elif usecase == "ChatBot with Tools":
                initial_state = {"messages": [user_message]}
                response = graph.invoke(initial_state)
                print(response)

                for message in response.get("messages", []):
                    if isinstance(message, HumanMessage):
                        self._display_message("user", message.content)

                    elif isinstance(message, ToolMessage):
                        # 🔧 Display tool execution + images if available
                        self._display_message("ai", message.content, expand_tool=True)

                    elif isinstance(message, AIMessage):
                        if message.content:
                            self._display_message("assistant", message.content)

                        if getattr(message, "tool_calls", None):
                            for tool_call in message.tool_calls:
                                tool_name = tool_call.get("name", "Unknown Tool")
                                with st.chat_message("assistant"):
                                    st.markdown(f"🛠️ **Tool Used:** `{tool_name}`")

            else:
                st.warning("⚠️ Unknown usecase selected. Please choose a valid option.")

        except Exception as e:
            st.error(f"🚨 Error processing user message: {str(e)}")
            st.stop()
