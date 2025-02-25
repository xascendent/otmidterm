import asyncio
import chainlit as cl
from langgraph.graph import StateGraph, END
from chains import (
    run_research_vector_store_node,
    run_research_llm_node,
    format_final_chain
)
from typing import Dict, TypedDict


# Define the state schema using TypedDict
class QueryState(TypedDict):
    user_query: str
    research_response: str
    final_response: str


async def supervisor_node(state: QueryState) -> Dict:
    """Decides where to route the query."""
    if state.get("final_response"):
        return {"next": "post_processing"}
    return {"next": "research"}


async def research_node(state: QueryState) -> QueryState:
    """Calls vector store first, then falls back to LLM if needed."""
    user_query = state["user_query"]

    research_response = await run_research_vector_store_node(user_query)

    if research_response == "NO DOCUMENT FOUND":
        research_response = await run_research_llm_node(user_query)

    state["research_response"] = research_response
    return state


async def post_processing_node(state: QueryState) -> QueryState:
    """Formats the final response before returning it."""
    formatted_response = format_final_chain.invoke(
        {"messages": [{"role": "system", "content": state["research_response"]}]}
    )

    state["final_response"] = formatted_response.content  # Update state
    return state  # Ensure state is returned correctly


# Define LangGraph with state schema
graph = StateGraph(QueryState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("research", research_node)
graph.add_node("post_processing", post_processing_node)

# Define edges (flow)
graph.add_edge("supervisor", "research")
graph.add_edge("research", "post_processing")
graph.add_edge("post_processing", END)

graph.set_entry_point("supervisor")

# Compile graph executor
research_graph_executor = graph.compile()


async def run_graph(user_query: str):
    """Run the LangGraph pipeline."""
    initial_state: QueryState = {"user_query": user_query, "research_response": "", "final_response": ""}

    async for output in research_graph_executor.astream(initial_state):
        final_state = output  # Capture the last state

        # If using Chainlit, yield intermediate state updates
        if "research_response" in final_state:
            await cl.Message(content=f"**Research Response:**\n{final_state['research_response']}").send()

    if "post_processing" in final_state and "final_response" in final_state["post_processing"]:
        return final_state["post_processing"]["final_response"]

    return "No response generated."


# Chainlit App
@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages in Chainlit."""
    if message.elements:  # This checks if a user tries to upload a file
        await cl.Message(content="❌ File uploads are not allowed.").send()
        return

    user_query = message.content
    response = await run_graph(user_query)

    await cl.Message(content=f"**Final Answer:**\n{response}").send()



if __name__ == "__main__":
    cl.run()
