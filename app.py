from typing import TypedDict, Annotated, List
from typing_extensions import List, TypedDict
from dotenv import load_dotenv
import chainlit as cl
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
import os
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from langchain_core.documents import Document
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd



load_dotenv()

# Load OpenAI Model
llm = ChatOpenAI(model="gpt-4o-mini")
qd_api_key = os.getenv("QDRANT_CLOUD_API_KEY")
EVALUATION_MODE = os.getenv("EVALUATION_MODE", "false").lower() == "false"


embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize Qdrant Client
qd_client = QdrantClient(
    "https://40c458f2-24a9-4153-b15b-0addf6a6bbcf.us-east-1-0.aws.cloud.qdrant.io:6333", 
    api_key=qd_api_key
)  

collection_name = "qt_document_collection"
hit_score = 0.5  # Minimum score for relevant hits

def search(query_vector, top_k=1) -> list:
    """Search for documents in Qdrant using an embedding vector."""
    print(f"🔍 Querying Qdrant with vector: {query_vector[:5]}...")  # Print first 5 elements for readability
    
    hits = qd_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )

    print(f"🛠 Raw Qdrant Response: {hits}")  # Print full response for debugging

    return_hits = [
        {"score": hit.score, "metadata": hit.payload}
        for hit in hits if hit.score > hit_score
    ]

    print(f"✅ Filtered Hits: {return_hits}")  # Show only the relevant results

    if not return_hits:
        print("⚠️ No relevant documents found in Qdrant.")
        return [{"score": 0, "metadata": "No relevant documents found."}]

    return return_hits

def evaluate_ragas_metrics(question: str, model_answer: str, retrieved_docs: list):
    """Evaluate faithfulness, context precision, and context recall using RAGAS."""
    
    # Extract document content from metadata
    ragas_docs = [
        Document(page_content=hit["metadata"].get("content", "")) 
        for hit in retrieved_docs if "content" in hit["metadata"] and hit["metadata"]["content"]
    ]

    if not ragas_docs:
        print("⚠️ No relevant documents to evaluate.")
        return {"faithfulness": 0, "context_precision": 0, "context_recall": 0}

    # Construct required input
    queries = [question]
    generated_answers = [model_answer]
    contexts = [[doc.page_content for doc in ragas_docs]]

    # Run evaluation
    scores = evaluate(
        queries=queries,
        contexts=contexts,
        generated_answers=generated_answers,
        metrics=[faithfulness, context_precision, context_recall]
    )

    print("📊 Debug: RAGAS Metrics Output ->", scores)

    # Extract individual scores
    faithfulness_score = scores.iloc[0]["faithfulness"]
    context_precision_score = scores.iloc[0]["context_precision"]
    context_recall_score = scores.iloc[0]["context_recall"]

    print(f"📊 Faithfulness Score: {faithfulness_score}")
    print(f"📊 Context Precision Score: {context_precision_score}")
    print(f"📊 Context Recall Score: {context_recall_score}")

    return {
        "faithfulness": faithfulness_score,
        "context_precision": context_precision_score,
        "context_recall": context_recall_score
    }

def evaluate_retrieved_docs(question: str, retrieved_docs: list):
    """Evaluate the retrieved documents using RAGAS metrics."""
    
    # Extract document content from metadata
    ragas_docs = [
        Document(page_content=hit["metadata"].get("content", ""))
        for hit in retrieved_docs
        if "content" in hit["metadata"] and hit["metadata"]["content"]
    ]

    # Debugging Output
    print("🔍 Debug: RAGAS Docs Format:", ragas_docs)

    if not ragas_docs:
        print("⚠️ No relevant documents to evaluate.")
        return 0  # Return low score if no documents found

    # Construct required input
    queries = [question]
    contexts = [[doc.page_content for doc in ragas_docs]]

    print("✅ Debug: Queries ->", queries)
    print("✅ Debug: Contexts ->", contexts)

    # Run evaluation
    scores = evaluate(
        queries=queries,
        contexts=contexts,
        metrics=[answer_relevancy]
    )

    print("📊 Debug: Raw Scores Output ->", scores)

    relevance_score = scores.iloc[0]["answer_relevancy"]
    print(f"📊 RAGAS Answer Relevancy Score: {relevance_score}")

    return relevance_score




def get_document_by_name(doc_name: str) -> str:
    """Retrieve the raw HTML content of a document by its name from the `data/` folder."""
    
    # Get the absolute path of the `data/` folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data")

    # Replace `.pdf` with `.html`
    html_doc_name = doc_name.replace(".pdf", ".html")
    full_path = os.path.join(data_path, html_doc_name)

    # Check if the file exists
    if not os.path.exists(full_path):
        print(f"⚠️ File not found: {full_path}")
        return "No file found"

    try:
        # Open and read the file content
        with open(full_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content  # Return the raw HTML content

    except Exception as e:
        print(f"❌ Error reading file {full_path}: {str(e)}")
        return "Error reading file"


# **User question prompt**
ot_user_prompt = """\
You are an occupational therapist providing accurate, evidence-based answers.
1. Only give correct information.
2. If unsure, respond with: "I don't know."
3. Be clear, concise, and helpful.

### Question
{question}

### Context
{context}
"""

# **Summarization prompt if a relevant document is found**
summarize_prompt = """\
Summarize the following document into a concise and easy-to-understand response for an occupational therapist.
Ensure the summary includes only relevant information.

### Document:
{document}
"""

# **Formatting post-processing prompt**
ot_formatted_prompt = """\
Given this data, format the response as follows:

1. **Eccentric Exercises**: 
2. **Isometric Exercises**: 
3. **Stretching**: 
4. **Manual Therapy**: 
5. **Ultrasound Therapy**: 
6. **Taping and Bracing**: 
7. **Functional Activities**: 
8. **Other**: 
9. **Document Title**: 
10. **Document File Name**: 

If a section does not have information, state: "I do not have information for this section." 
If Document Title and Document File Name are missing, remove sections 9 and 10.

### Context
{context}
"""

# **Create ChatPromptTemplate**
rag_prompt = ChatPromptTemplate.from_template(ot_user_prompt)
summary_prompt = ChatPromptTemplate.from_template(summarize_prompt)
format_prompt = ChatPromptTemplate.from_template(ot_formatted_prompt)

# **Research Node: Queries Qdrant first, then the LLM if needed**
def research_node(state) -> dict:
    question = state["messages"][-1].content

    # Convert the text question to an embedding using OpenAI Embeddings
    query_vector = embedding_model.embed_query(question)

    # Query Qdrant with the vector
    relevant_docs = search(query_vector=query_vector, top_k=1) 

    model_answer = "No answer generated yet"

    if relevant_docs[0]['score'] > hit_score:  # Threshold for good retrieval quality this will be the cosine similarity score
        # Found relevant document → Summarize it
        document_name = relevant_docs[0]["metadata"].get("document_name", "No source available.")
        document_text = get_document_by_name(document_name)
        messages = summary_prompt.format_messages(document=document_text)
        response = llm.invoke(messages)

        if EVALUATION_MODE:
            # Evaluate retrieved documents using RAGAS
            relevance_score = evaluate_retrieved_docs(question, relevant_docs)        
            print(f"📊 [Evaluation Mode] RAGAS Score: {relevance_score}")
            ragas_scores = evaluate_ragas_metrics(question, model_answer, relevant_docs)
            print(f"📊 [evaluate_ragas_metrics] RAGAS Scores: {ragas_scores}")

        return {**state, "messages": state["messages"] + [HumanMessage(content=response.content)], "_next": "post_processing"}
    
    else:
        # No relevant document        
        messages = rag_prompt.format_messages(question=question, context="No relevant documents found.")
        response = llm.invoke(messages)

        return {**state, "messages": state["messages"] + [HumanMessage(content=response.content)], "_next": "post_processing"}

def compare_text_similarity(text1, text2):
    """Compute cosine similarity between two texts using embeddings."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    emb1 = np.array(embeddings.embed_query(text1)).reshape(1, -1)
    emb2 = np.array(embeddings.embed_query(text2)).reshape(1, -1)

    return cosine_similarity(emb1, emb2)[0][0]  # Return similarity score

def evaluate_against_golden_set(question, model_answer):
    """Compare model-generated answers against the golden dataset and display results in a DataFrame."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data")    
    full_path = os.path.join(data_path, "testingset.json")
   
    if not os.path.exists(full_path):
        print(f"❌ Error: Golden dataset not found at {full_path}")
        return None

    # Load JSON Data
    with open(full_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    # Store results in a list for Pandas DataFrame
    results = []

    for entry in golden_data:
        expected_answer = entry.get("expected_answer", "").strip()
        q = entry.get("question", "").strip()

        # Compute similarity score
        similarity_score = compare_text_similarity(model_answer, expected_answer)

        # Append to results list
        results.append({
            "Question": q,
            "Expected Answer": expected_answer,
            "Model Answer": model_answer,
            "Similarity Score": round(similarity_score, 2)  # Round to 2 decimal places
        })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Print DataFrame
    print("\n📊 **Evaluation Results**")
    print(df.to_string(index=False))  # Pretty-print without row index

    return df



# **Post-Processing Node: Formats response using `ot_formatted_prompt`**
def post_processing_node(state) -> dict:
    response_text = state["messages"][-1].content
    
    # Evaluate the model against the golden dataset
    if EVALUATION_MODE:
        question = state["messages"][0].content
        pdf = evaluate_against_golden_set(question, response_text)

    messages = format_prompt.format_messages(context=response_text)
    response = llm.invoke(messages)

    return {**state, "messages": state["messages"] + [HumanMessage(content=response.content)]}

# **Supervisor Node: Directs to research node**
def supervisor_node(state) -> dict:
    return {**state, "_next": "research"}

# **Define LangGraph workflow**
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: List[Document]

uncompiled_graph = StateGraph(AgentState)

uncompiled_graph.add_node("supervisor", supervisor_node)
uncompiled_graph.add_node("research", research_node)
uncompiled_graph.add_node("post_processing", post_processing_node)

uncompiled_graph.add_edge("supervisor", "research")
uncompiled_graph.add_edge("research", "post_processing")
uncompiled_graph.add_edge("post_processing", END)

uncompiled_graph.set_entry_point("supervisor")

# **Compile graph executor**
research_graph_executor = uncompiled_graph.compile()

### **Chainlit Integration**
@cl.on_chat_start
async def start():
    cl.user_session.set("graph", research_graph_executor)

@cl.on_message
async def handle(message: cl.Message):
    graph = cl.user_session.get("graph")
    state = {"messages": [HumanMessage(content=message.content)], "context": []}  
    response = await graph.ainvoke(state)
    await cl.Message(content=response["messages"][-1].content).send()
