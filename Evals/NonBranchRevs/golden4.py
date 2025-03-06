import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from uuid import uuid4
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_core.prompts import ChatPromptTemplate
from ragas.testset import TestsetGenerator
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    LLMContextRecall, Faithfulness, FactualCorrectness, 
    ResponseRelevancy, ContextEntityRecall, NoiseSensitivity
)
from ragas.evaluation import RunConfig
from ragas import EvaluationDataset
import ragas
from langsmith import Client
from langsmith.evaluation import LangChainStringEvaluator 
from langsmith.evaluation import evaluate as langsmith_evaluate

os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "RAG-Comparison"

# Load environment variables
load_dotenv()

# Constants
MODEL = "gpt-4o"   # "gpt-3.5-turbo"  # "gpt-4o"  #  

# Embedding model options
EMBEDDING_MODELS = {
    "openai-large": "text-embedding-3-large",
    "openai-small": "text-embedding-3-small",
    "huggingface": "shivXy/ot-midterm-v0"
}

# Select which embedding model to use
EMBEDDING_CHOICE = "huggingface"  # Change this to "openai-large", "openai-small", or "huggingface"
EMBEDDINGS_MODEL = EMBEDDING_MODELS[EMBEDDING_CHOICE]

DATA_PATH = "data/"
CACHE_PATH = f"ot_golden_dataset_{EMBEDDING_CHOICE}.csv"  # Separate cache files for different embedding models
EVAL_PATH = f"ot_complete_evaluation_{EMBEDDING_CHOICE}.csv"  # Separate evaluation files for different embedding models
client = Client()

os.environ["LANGCHAIN_TAGS"] = f"{MODEL}_{EMBEDDING_CHOICE}"

RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""

def get_embeddings_model():
    """Returns the appropriate embeddings model based on the selected choice."""
    if EMBEDDING_CHOICE.startswith("openai"):
        return OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
    elif EMBEDDING_CHOICE == "huggingface":
        return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    else:
        raise ValueError(f"Unknown embedding model choice: {EMBEDDING_CHOICE}")

def load_documents() -> list:
    """Loads HTML documents from the specified directory."""
    text_loader = DirectoryLoader(DATA_PATH, glob="*.html", loader_cls=BSHTMLLoader)
    documents = text_loader.load()
    print(f"Loaded {len(documents)} documents from {DATA_PATH}")
    return documents

def create_vector_store(documents):
    """Creates a vector store from the loaded documents."""
    embeddings = get_embeddings_model()
    vectorstore = Qdrant.from_documents(
        documents,
        embeddings,
        location=":memory:",  # In-memory storage for testing
        collection_name="OT"
    )
    return vectorstore

def clean_context(context):
    """Removes excessive newlines from a context string."""
    return " ".join(context.split())  # Removes excessive spaces and newlines

def generate_golden_dataset(documents, testset_size=10, save_path=CACHE_PATH):
    """Generates a golden dataset from the loaded documents."""
    print("Generating new golden dataset...")
    
    # Setup LLM and embeddings for RAGAS
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model=MODEL))
    generator_embeddings = LangchainEmbeddingsWrapper(get_embeddings_model())
    
    # Create test set generator
    generator = TestsetGenerator(
        llm=generator_llm, 
        embedding_model=generator_embeddings
    )
    
    # Generate test set
    testset = generator.generate_with_langchain_docs(
        documents, 
        testset_size=testset_size
    )
    
    # Convert to pandas
    df = testset.to_pandas()
    if EMBEDDING_MODELS[EMBEDDING_CHOICE] != "shivXy/ot-midterm-v0":
        df["contexts"] = df["contexts"].apply(lambda ctx: [clean_context(c) for c in ctx] if isinstance(ctx, list) else clean_context(ctx))

    # Save dataset
    df.to_csv(save_path, index=False)
    print(f"Golden dataset with {len(df)} examples saved to {save_path}")
    
    return df

def generate_rag_answers(golden_df, vectorstore):
    """Generate RAG-based answers for the golden dataset questions."""
    print("Generating answers using RAG system...")
    
    # Setup RAG components
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    llm = ChatOpenAI(model=MODEL)
    
    answers = []
    
    # Using the correct column name 'user_input' for questions
    for _, row in golden_df.iterrows():
        question = row['user_input']
        
        # Retrieve relevant documents
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate answer
        chain = rag_prompt | llm
        answer = chain.invoke({"question": question, "context": context})
        answers.append(answer.content)
    
    # Add answers to the dataframe
    golden_df['answer'] = answers
    
    # Also store the retrieved contexts for evaluation
    golden_df['retrieved_contexts'] = golden_df.apply(
        lambda row: vectorstore.similarity_search(row['user_input'], k=3),
        axis=1
    )
    
    # Convert document objects to strings
    golden_df['contexts'] = golden_df['retrieved_contexts'].apply(
        lambda docs: [doc.page_content for doc in docs]
    )
    if EMBEDDING_MODELS[EMBEDDING_CHOICE] != "shivXy/ot-midterm-v0":
        golden_df["contexts"] = golden_df["contexts"].apply(lambda ctx: [clean_context(c) for c in ctx] if isinstance(ctx, list) else clean_context(ctx))
    
    return golden_df

def evaluate_rag_performance(complete_df):
    """Evaluates the RAG system using RAGAS metrics."""
    print("Evaluating RAG performance with RAGAS...")
    
    # Get the data from the dataframe
    questions = complete_df['user_input'].tolist()
    contexts = complete_df['contexts'].tolist()
    answers = complete_df['answer'].tolist()
    ground_truths = complete_df['reference'].tolist()
    
    # Setup evaluator
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=MODEL))
    run_config = RunConfig(timeout=120)
    
    # Load the evaluation dataset using individual arrays
    from datasets import Dataset
    
    # Create a dict with the required format
    eval_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_dict(eval_dict)
    
    # Run evaluation
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            FactualCorrectness(),
            ResponseRelevancy(),
            ContextEntityRecall()
        ],
        llm=evaluator_llm,
        run_config=run_config
    )
    
    print("Evaluation results:")
    print(result)
    return result

def langsmith_create_evaluations(complete_df):
    """Create and run evaluations in LangSmith using the RAG dataset."""
    # Set up the RAG chain that will be evaluated
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    llm = ChatOpenAI(model=MODEL)
    chain = rag_prompt | llm
    
    client = Client()
    # dataset_name = f"OT EVALS - {EMBEDDING_CHOICE}"  # Include embedding choice in dataset name
    dataset_name = "OT EVALS"

    # Create the dataset in LangSmith
    try:
        # Try to get existing dataset first
        langsmith_dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Using existing dataset: {dataset_name}")
    except:
        # Create new dataset if it doesn't exist
        langsmith_dataset = client.create_dataset(
            dataset_name=dataset_name,
            #description=f"OT EVALS with {EMBEDDING_CHOICE} embeddings"
            description="OT EVALS"
        )
        print(f"Created new dataset: {dataset_name}")

    # Clear existing examples to avoid duplicates
    try:
        examples = client.list_examples(dataset_id=langsmith_dataset.id)
        for example in examples:
            client.delete_example(example_id=example.id)
        print(f"Cleared existing examples from dataset")
    except:
        print("No existing examples to clear")

    # Add examples to the dataset with proper structure
    for _, row in complete_df.iterrows():
        # Convert context lists to string if needed
        context = row["contexts"]
        if isinstance(context, list):
            context = "\n\n".join(context)
            
        client.create_example(
            inputs={
                "question": row["user_input"],
                "context": context  # Include context in inputs matching the chain's expected input
            },
            outputs={
                "answer": row["reference"]  # The ground truth answer
            },
            dataset_id=langsmith_dataset.id
        )

    print(f"Created LangSmith dataset with {len(complete_df)} examples")
    
    # Define the function that will process each example
    def run_chain(inputs):
        result = chain.invoke({
            "question": inputs["question"],
            "context": inputs["context"]
        })
        return {"response": result.content}  # Return content from ChatOpenAI
    
    # Set up the evaluator
    eval_llm = ChatOpenAI(model=MODEL)
    qa_evaluator = LangChainStringEvaluator(
        "qa", 
        config={"llm": eval_llm},
        prepare_data=lambda run, example: {
            "prediction": run.outputs["response"],  # Match the key in run_chain's return
            "reference": example.outputs["answer"],
            "input": example.inputs["question"],
        }
    )
    
    # Using the original langsmith_evaluate call format from your code
    # This matches the import: from langsmith.evaluation import evaluate as langsmith_evaluate
    print("Running LangSmith evaluations...")
    try:
        eval_results = langsmith_evaluate(
            run_chain,  # This is the target function
            data=dataset_name,
            evaluators=[qa_evaluator],
            metadata={"model": MODEL, "embeddings": EMBEDDINGS_MODEL, "embedding_type": EMBEDDING_CHOICE}
        )
        print(f"Completed LangSmith evaluations: {eval_results}")
        return eval_results
    except TypeError as e:
        # Let's try the client.run_evaluation format as an alternative
        print(f"Error with langsmith_evaluate: {e}")
        print("Trying alternative approach with client.run_evaluation...")
        
        # Alternative approach
        evaluation_run = client.run_evaluation(
            project_name=os.environ["LANGCHAIN_PROJECT"],
            evaluation_name=f"OT RAG Evaluation - {EMBEDDING_CHOICE}",
            dataset_name=dataset_name,
            llm_or_chain_factory=run_chain,
            evaluators=[qa_evaluator],
            metadata={"model": MODEL, "embeddings": EMBEDDINGS_MODEL, "embedding_type": EMBEDDING_CHOICE}
        )
        print(f"Started evaluation run: {evaluation_run}")
        return evaluation_run


def main():
    print(f"Using embedding model: {EMBEDDING_CHOICE} ({EMBEDDINGS_MODEL})")
    
    # Load documents
    documents = load_documents()
    
    # Create vector store
    vectorstore = create_vector_store(documents)
    
    # Generate golden dataset or load if exists
    if os.path.exists(CACHE_PATH):
        print(f"Loading existing golden dataset from {CACHE_PATH}")
        golden_df = pd.read_csv(CACHE_PATH)
    else:
        golden_df = generate_golden_dataset(documents)
    
    # Generate answers using your RAG system
    complete_df = generate_rag_answers(golden_df, vectorstore)
    
    # Save complete dataset with answers
    complete_df.to_csv(EVAL_PATH, index=False)
    print(f"Complete evaluation dataset saved to {EVAL_PATH}")
    
    # Evaluate RAG performance
    evaluation_results = evaluate_rag_performance(complete_df)
    print(complete_df) #    user_input  ...                                           contexts
    print(evaluation_results) # prints {'faithfulness': 0.5065, 'factual_correctness': 0.3018, 'answer_relevancy': 0.4252, 'context_entity_recall': 0.5344}
    # append the evaluation results to a file
    with open("evaluation_results.txt", "a") as f:
        f.write(f"'Model Used': '{MODEL}' 'Embeddings Model Used': '{EMBEDDING_CHOICE}/{EMBEDDINGS_MODEL}' 'RESULTS':{evaluation_results}\n")

    langsmith_create_evaluations(complete_df)
    print("Done.")

if __name__ == "__main__":
    main()