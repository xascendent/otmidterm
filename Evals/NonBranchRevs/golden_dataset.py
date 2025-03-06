import nest_asyncio
import os
from dotenv import load_dotenv
import pandas as pd
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain_community.vectorstores import Qdrant
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate, RunConfig
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall,
    NoiseSensitivity
)
from langsmith import Client
from langsmith.evaluation import LangChainStringEvaluator, evaluate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Constants
MODEL = "gpt-3.5-turbo"
EMBEDDINGS_MODEL = "text-embedding-3-small"
DATA_PATH = "data/"

def load_documents() -> list:
    """Loads HTML documents from the specified directory."""
    text_loader = DirectoryLoader(DATA_PATH, glob="*.html", loader_cls=BSHTMLLoader)
    documents = text_loader.load()
    return documents

def load_vector_store(documents):
    """Creates a vector store from the loaded documents."""
    embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
    vectorstore = Qdrant.from_documents(
        documents,
        embeddings,
        location=":memory:",  # In-memory storage for testing
        collection_name="OT"
    )
    return vectorstore

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=MODEL))

def create_retriever_test_set(retriever, dataset):
    for test_row in dataset:
        response = retriever.invoke({"query": test_row.eval_sample.user_input})
        test_row.eval_sample.response = response["result"]
        test_row.eval_sample.retrieved_contexts = [context.page_content for context in response["source_documents"]]
    return dataset

custom_run_config = RunConfig(timeout=360)

def perform_ragas_eval(evaluation_dataset, run_config=custom_run_config):
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            LLMContextRecall(),
            Faithfulness(),
            FactualCorrectness(),
            ResponseRelevancy(),
            ContextEntityRecall(),
            NoiseSensitivity()
        ],
        llm=evaluator_llm,
        run_config=custom_run_config
    )
    return result

# Define evaluator for LangSmith
qa_evaluator = LangChainStringEvaluator(
    "qa",
    config={"llm": ChatOpenAI(model=MODEL)},  # Use direct ChatOpenAI instance
    prepare_data=lambda run, example: {
        "prediction": run.outputs["response"],
        "reference": example.outputs["answer"],
        "input": example.inputs["question"],
    }
)

def invoke_with_correct_inputs(chain, example):
    """Ensure correct key access for question input."""
    print(f"Example structure: {example}")  # Debugging step

    # If 'inputs' is not present, fallback to first key found
    if isinstance(example, dict) and "inputs" in example:
        inputs_data = example["inputs"]
    else:
        inputs_data = example  # Fallback if 'inputs' is missing

    # Extract the question key safely
    question_key = "question" if "question" in inputs_data else list(inputs_data.keys())[0]
    
    return chain.invoke({"query": inputs_data.get(question_key, "")})


def langsmith_evaluate(chain, test_set):
    """Evaluates the model using LangSmith dataset, ensuring correct format."""
    
    # Convert test_set to a list of dictionaries
    dataset = []
    for sample in test_set:
        dataset.append({
            "inputs": {"question": sample.eval_sample.user_input},  # Extract user question
            "outputs": {"response": sample.eval_sample.reference},  # Expected response
        })

    print("First few dataset entries for debugging:")
    for item in dataset[:5]:  # Print first few items to inspect structure
        print(item)

    evaluate(
        lambda example: invoke_with_correct_inputs(chain, example),  # Call global function
        data=dataset,  # Use formatted dataset
        evaluators=[qa_evaluator],
        metadata={"revision_id": "default_chain_init"},
    )


from langchain_core.prompts import ChatPromptTemplate

RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI()


def main():
    print("Loading documents...")
    documents = load_documents()

    if not documents:
        print("No documents loaded. Check your data path and file format.")
        return

    print("Creating vector store...")
    vectorstore = load_vector_store(documents)

    # Set up retriever
    naive_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Set up the RAG chain
    llm = ChatOpenAI(model=MODEL)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=naive_retriever,
        return_source_documents=True
    )

    # Ask a question
    query = "How should an occupational therapist treat a hand burn?"
    result = rag_chain.invoke({"query": query})

    print("\n### Answer:\n", result["result"])
    print("\n### Retrieved Documents:\n", result["source_documents"])

    naive_retrieval_chain = (
    # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
    # "question" : populated by getting the value of the "question" key
    # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
    {"context": itemgetter("question") | naive_retriever, "question": itemgetter("question")}
    # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
    #              by getting the value of the "context" key from the previous step
    | RunnablePassthrough.assign(context=itemgetter("context"))
    # "response" : the "context" and "question" values are used to format our prompt object and then piped
    #              into the LLM and stored in a key called "response"
    # "context"  : populated by getting the value of the "context" key from the previous step
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)






    # Generate test dataset
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model=MODEL))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)    
    dataset = generator.generate_with_langchain_docs(documents, testset_size=10)
    
    if isinstance(dataset, dict):  # Ensure correct format
        df = pd.DataFrame.from_dict(dataset)
    else:
        df = dataset.to_pandas()

    dataset.upload() # Upload to RAGAS

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=MODEL))

    
    for test_row in dataset:
        response = rag_chain.invoke({"question" : test_row.eval_sample.user_input})
        test_row.eval_sample.response = response["response"].content
        test_row.eval_sample.retrieved_contexts = [context.page_content for context in response["context"]]
    



    print(df)
    df.to_csv("golden_dataset.csv", index=False)

    # Create test set for retriever
    test_set = create_retriever_test_set(rag_chain, dataset)
    ts = test_set.to_pandas()

    print("\n### Test Set:\n", ts)

    # Upload to LangSmith
    client = Client()

    dataset_name = "OT_dataset"

    # Try to retrieve dataset; if it doesn't exist, create it
    try:
        langsmith_dataset = client.read_dataset(name=dataset_name)
    except Exception:
        langsmith_dataset = client.create_dataset(dataset_name=dataset_name, description="OT dataset")

    dataset_id = langsmith_dataset.id

# Upload test set examples to LangSmith
    for row in ts.itertuples():
        client.create_example(
            inputs={"question": row.user_input},  # Ensure query-like input
            outputs={"response": row.reference},  # Fix output key
            metadata={"context": row.reference_contexts},
            dataset_id=dataset_id
        )
    print(f"Dataset successfully uploaded. Dataset Name: {dataset_name}, Dataset ID: {dataset_id}")

    # Run evaluation
    langsmith_evaluate(rag_chain, test_set)

if __name__ == "__main__":
    main()

