from typing import TypedDict, Annotated, List
from typing_extensions import List
from dotenv import load_dotenv
import chainlit as cl
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models
import os
from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader

# Chronic lateral elbow tendinopathy with a supervised graded exercise protocol

# Define the state structure
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context: List[Document]    

MODEL = "gpt-4o"
DATA_PATH = "data/"

# Initialize the OpenAI LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
)

openai_chat_model = ChatOpenAI(model=MODEL)

qd_api_key = os.getenv("QDRANT_CLOUD_API_KEY")
qd_client = QdrantClient(
    "https://40c458f2-24a9-4153-b15b-0addf6a6bbcf.us-east-1-0.aws.cloud.qdrant.io:6333", 
    api_key=qd_api_key
)  
collection_name = "qt_document_collection"


# RAG prompt template
# Prompts
ot_user_prompt = """\
You are an occupational therapist providing accurate, evidence-based answers.
1. Only give correct information.
2. If unsure, respond with: "I don't know."
3. Be clear, concise, and helpful.

### Question
{question}

"""

gen_prompt = """
### Question
{question}
"""

ot_question_prompt = """
Could this question be a occupational therapy question? If so, say "Yes" otherwise say "No".

### Question
{question}
"""

ot_ext_answer_prompt = """\
You are an occupational therapist delivering accurate, evidence-based answers. You have received data from a previous search. Now, incorporate any additional relevant information from the knowledge base related to the user's question.

### Question
{question}

### Context
{context}
"""

summarize_prompt = """\
Summarize the following document into a concise and easy-to-understand response for an occupational therapist.
Ensure the summary includes only relevant information.  We will be looking for these key points.  They are Eccentric Exercises, Isometric Exercises, Stretching, Manual Therapy, Ultrasound Therapy, Taping and Bracing, Functional Activities.

### Document:
{document}
"""

ot_formatted_prompt = """\
Given this data, format the response as follows.  The **Other** section might already be in the context just format it correctly:

1. **Eccentric Exercises**: 
2. **Isometric Exercises**: 
3. **Stretching**: 
4. **Manual Therapy**: 
5. **Ultrasound Therapy**: 
6. **Taping and Bracing**: 
7. **Functional Activities**: 
8. **Other**: 

### Context
{context}
"""

# Create ChatPromptTemplate
rag_prompt = ChatPromptTemplate.from_template(ot_user_prompt)
summary_prompt = ChatPromptTemplate.from_template(summarize_prompt)
format_prompt = ChatPromptTemplate.from_template(ot_formatted_prompt)
other_prompt = ChatPromptTemplate.from_template(ot_ext_answer_prompt)
ot_question_prompt = ChatPromptTemplate.from_template(ot_question_prompt)
gn_prompt = ChatPromptTemplate.from_template(gen_prompt)

no_data_found_chain = rag_prompt | openai_chat_model 
data_found_chain = summary_prompt | openai_chat_model
other_info_about_topic = other_prompt | openai_chat_model
format_content = format_prompt | openai_chat_model
is_ot_question = ot_question_prompt | openai_chat_model
no_ot_question = gn_prompt | openai_chat_model

# Function to retrieve relevant documents
def retrieve_context(query: str, top_k: int = 3):
    """Retrieve relevant documents from the vectorstore."""
    #docs = vectorstore.similarity_search(query, k=top_k)
    #return docs

def supervisor_node(state) -> dict:
    return {**state, "_next": "research"}


# Define our function to call the LLM
def research_node(state):
    """Call the LLM with the current messages and return a dictionary."""
    messages = state["messages"]
    # Extract user query
    user_query = state["messages"][-1].content

    is_ot = is_ot_question.invoke({"question": user_query})
    if is_ot.content == "No":
        response = no_ot_question.invoke({"question": user_query})
        return {"messages": messages + [response.content], "context": state.get("context", [])}
        




    # Perform a metadata search instead of vector search
    hits = qd_client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="title",  # Change this to the actual field name storing text
                    match=models.MatchValue(value=user_query)
                )
            ]
        ),
        limit=1        
    )

    # Check if any data was found
    retrieved_docs = [hit.payload for hit in hits[0]]  # hits[0] contains the documents
    final_message = ""

    # Print or return the results
    #print(retrieved_docs) #[{'document_name': 'Tennis elbow graded exercise.html', 'document_id': '85152085-8c16-4fe9-9302-ad94f260364e', 'document_date': '2025-02-24', 'title': 'Chronic lateral elbow tendinopathy with a supervised graded exercise protocol', 'chunk_number': 1, 'description': 'No description Found', 'author': 'Arzu Razak &#x00D6;zdin&#x00E7;ler PT, PhD', 'tags': ['tag1', 'tag2', 'tag3'], 'subject': 'Journal of Hand Therapy, 36 (2023) 913-922. doi:10.1016/j.jht.2022.11.005'}]
    summerized_response = ""
    if retrieved_docs:
        for doc in retrieved_docs:
            doc_name = os.path.splitext(doc['document_name'])[0] + ".html"
            text_loader = DirectoryLoader(DATA_PATH, glob=doc_name, loader_cls=BSHTMLLoader)
            doc_context = text_loader.load()  # Load the document content
            temp_summerized_response = data_found_chain.invoke({"document": doc_context})
            summerized_response = summerized_response + temp_summerized_response.content
            #print(doc_name) 
            #print(document) # he pain in- \ntensity was reduced as a result of the basic exercise protocol.\n\n\nMaterials and methods \nA prospective case series study was conducted on patients with \nLET. Patients were referred to the physiotherapy unit of the au- \nthorsâ€™ institution from the Department of Orthopedics and Trau- \nmatology of a Univers
        
        final_response = other_info_about_topic.invoke({"question": user_query, "context": summerized_response})
        final_message = summerized_response + " **OTHER**: " + final_response.content
        
    else:
        print("No relevant documents found for this query.")
        response = no_data_found_chain.invoke({"question": user_query}) 
        final_message = response.content
    
    # Return updated state
    return {"messages": messages + [final_message], "context": state.get("context", [])}

def post_processing_node(state):
    messages = state["messages"][-1]

    # see if this is a OT question if not we will just return the last response otherwise we will format the response
    is_ot = is_ot_question.invoke({"question": messages.content})
    if is_ot.content == "No":
        return state
    
    # Extract content from the message
    response_text = messages.content if hasattr(messages, 'content') else messages
    
    # Invoke the formatting LLM
    formatted_response = format_content.invoke({"context": response_text})
    
    # Ensure it's stored as an AIMessage
    formatted_message = AIMessage(content=formatted_response.content)
    
    # Return updated state
    return {"messages": state["messages"] + [formatted_message], "context": state.get("context", [])}


# Create the graph
graph = StateGraph(AgentState)

# Add the LLM node

graph.add_node("supervisor", supervisor_node)
graph.add_node("research", research_node)
graph.add_node("post_processing", post_processing_node)

# Set the entry point
graph.set_entry_point("supervisor")

# Define the simple flow
graph.add_edge("supervisor", "research")
graph.add_edge("research", "post_processing")
graph.add_edge("post_processing", END)

# Compile the graph
compiled_graph = graph.compile()

@cl.on_chat_start
async def start():
    cl.user_session.set("graph", compiled_graph)

@cl.on_message
async def handle(message: cl.Message):
    graph = cl.user_session.get("graph")
    
    # Initialize state with the user's message
    state = {"messages": [HumanMessage(content=message.content)], "context": []}
    
    # Invoke the graph with the current state
    response = await graph.ainvoke(state)
    
    # Send the AI's response back to the user
    await cl.Message(content=response["messages"][-1].content).send()