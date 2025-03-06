import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
import numpy as np


os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

#print(ragas.__version__)
#print(Client.__version__)
# Load environment variables
load_dotenv()

# Constants
MODEL = "gpt-3.5-turbo"
EMBEDDINGS_MODEL = "text-embedding-3-small"
DATA_PATH = "data/"
CACHE_PATH = "ot_golden_dataset.csv"
EVAL_PATH = "ot_complete_evaluation.csv"



os.environ["LANGCHAIN_TAGS"] = MODEL

RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""

def load_documents() -> list:
    """Loads HTML documents from the specified directory."""
    text_loader = DirectoryLoader(DATA_PATH, glob="*.html", loader_cls=BSHTMLLoader)
    documents = text_loader.load()
    print(f"Loaded {len(documents)} documents from {DATA_PATH}")
    return documents

def create_vector_store(documents):
    """Creates a vector store from the loaded documents."""
    embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
    vectorstore = Qdrant.from_documents(
        documents,
        embeddings,
        location=":memory:",  # In-memory storage for testing
        collection_name="OT"
    )
    return vectorstore

def generate_golden_dataset(documents, testset_size=10, save_path=CACHE_PATH):
    """Generates a golden dataset from the loaded documents."""
    print("Generating new golden dataset...")
    
    # Setup LLM and embeddings for RAGAS
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model=MODEL))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBEDDINGS_MODEL))
    
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
 



import numpy as np
import pandas as pd
from uuid import uuid4
from langsmith import Client



import pandas as pd
from langsmith import Client

def save_ragas_results_to_csv(evaluation_results, csv_path):
    """Converts a RAGAS EvaluationDataset to a CSV file for LangSmith upload."""
    
    # Convert EvaluationDataset to Pandas DataFrame
    dataset_df = evaluation_results.dataset.to_pandas()

    # Debugging: Print dataset structure
    print("Dataset Columns:", dataset_df.columns)
    print("Dataset Preview:\n", dataset_df.head())

    # Standardize column names to match expected format
    column_mapping = {
        "user_input": "question",
        "retrieved_contexts": "contexts",
        "response": "answer",
        "reference": "ground_truth"
    }
    dataset_df = dataset_df.rename(columns=column_mapping)

    # Ensure missing values are handled properly
    dataset_df = dataset_df.fillna("")

    # Add evaluation scores (metrics) from RAGAS results
    if hasattr(evaluation_results, "scores"):
        metrics_dict = evaluation_results.scores  # Extract scores
        if isinstance(metrics_dict, dict):  # Ensure it is a dictionary
            for metric_name, metric_values in metrics_dict.items():
                dataset_df[metric_name] = metric_values

    # Save the processed dataset to CSV
    dataset_df.to_csv(csv_path, index=False)
    print(f"Evaluation results saved to {csv_path}")

def upload_csv_to_langsmith(csv_path):
    """Uploads a CSV file to LangSmith."""
    client = Client()

    input_keys = ["question", "contexts"]  # Adjust to match the dataset
    output_keys = ["answer", "ground_truth"]  # Adjust to match expected outputs

    dataset = client.upload_csv(
        csv_file=csv_path,
        input_keys=input_keys,
        output_keys=output_keys,
        name="RAG Evaluation Dataset",
        description="Evaluation dataset for RAG performance tracking",
        data_type="kv",
    )

    print(f"Successfully uploaded dataset to LangSmith: {dataset.id}")


from langsmith import Client
import pandas as pd
from langsmith.evaluation import LangChainStringEvaluator, evaluate


def langsmith_evaluate(csv_path, chain, dataset_name):
    client = Client()
    dataset_name = "OT EVALS"

    langsmith_dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="OT EVALS"
    )

    # Load the dataset
    df = pd.read_csv(csv_path)
    for data_row in df.iterrows():
            client.create_example(
                inputs={
                    "question": data_row[1]["user_input"]
                },
                outputs={
                    "answer": data_row[1]["reference"]
                },
                metadata={
                    "context": data_row[1]["reference_contexts"]
                },
                dataset_id=langsmith_dataset.id
            )

    eval_llm = ChatOpenAI(model=MODEL)
    qa_evaluator = LangChainStringEvaluator("qa", config={"llm" : eval_llm},
                                            prepare_data=lambda run, example: {
                                                "prediction": run.outputs["response"],
                                                "reference": example.outputs["answer"],
                                                "input": example.inputs["question"],
                                            })
    evaluate(
        chain.invoke,
        data=dataset_name,
        evaluators=[qa_evaluator],
        metadata={"revision_id": "default_chain_init"},
    )










def run_langsmith_evaluation(csv_path):
    """Runs an evaluation using LangSmith by logging evaluation runs manually."""
    client = Client()

    # Load the dataset
    df = pd.read_csv(csv_path)

    # Define evaluation metadata
    evaluation_config = {
        "model": "RAG",
        "tags": ["RAG", "Evaluation"],
    }

    

    # Log each row as an evaluation in LangSmith
    for i, row in df.iterrows():
        client.create_run(
            name=f"RAG_Evaluation_{uuid4().hex[:8]}",  # Unique run name
            run_type="chain",
            inputs=row.to_dict(),  # Log the dataset row as input
            outputs={},  # Optionally log outputs if applicable
            extra={"metrics": evaluation_config["model"]},  # Include model name in metadata
            tags=evaluation_config["tags"],  # Attach evaluation tags
        )

    print("✅ Successfully logged evaluations to LangSmith Experiments.")



def upload_to_langsmith_evaluation(evaluation_results, model_name):
    """Uploads the evaluation results to LangSmith for structured evaluation tracking."""
    try:
        client = Client()  # Ensure LangSmith client is initialized
        dataset_name = "RAG_Evaluation_Dataset"  # Define a dataset to store results

        # Extract scores properly from evaluation_results
        metrics_dict = evaluation_results.scores  # This contains the computed metrics

        # Debugging: Print structure of metrics_dict
        print("Metrics Dict Type:", type(metrics_dict))
        print("Metrics Dict Content:", metrics_dict)

        # Convert dataset to Pandas DataFrame for easier access
        dataset_df = evaluation_results.dataset.to_pandas()

        # Debugging: Print dataset structure
        print("Columns in dataset_df:", dataset_df.columns)
        print("Dataset preview:\n", dataset_df.head())

        # Ensure NaN values are handled properly
        dataset_df = dataset_df.fillna("")  # Replace NaN with empty string to avoid JSON serialization issues

        # Standardize column names (strip spaces and convert to lowercase)
        dataset_df.columns = dataset_df.columns.str.strip().str.lower()

        # Define column mapping based on actual dataset
        column_mapping = {
            "user_input": "question",
            "retrieved_contexts": "contexts",
            "response": "answer",
            "reference": "ground_truth"
        }

        # Rename columns to match expected format
        dataset_df = dataset_df.rename(columns=column_mapping)

        # Expected columns
        required_columns = {"question", "answer", "contexts", "ground_truth"}
        missing_columns = required_columns - set(dataset_df.columns)

        if missing_columns:
            raise ValueError(f"Missing required columns in dataset: {missing_columns}")

        # Convert evaluation results into a structured format
        eval_data = []
        for i, row in dataset_df.iterrows():
            entry = {
                "question": row["question"],
                "answer": row["answer"],
                "context": row["contexts"],
                "ground_truth": row["ground_truth"],
            }

            # Handle both dictionary and list cases for metrics_dict
            if isinstance(metrics_dict, dict):
                for metric in metrics_dict.keys():
                    entry[metric] = (
                        float(metrics_dict[metric][i]) if i < len(metrics_dict[metric]) else 0.0
                    )
            elif isinstance(metrics_dict, list):
                for metric_entry in metrics_dict:
                    if isinstance(metric_entry, dict):
                        for metric, value in metric_entry.items():
                            entry[metric] = float(value) if isinstance(value, (int, float, np.float64)) else 0.0

            eval_data.append(entry)

        # Create or reference an evaluation dataset in LangSmith
        try:
            client.create_dataset(dataset_name=dataset_name)
        except Exception:
            pass  # The dataset likely already exists

        # Log each evaluation entry separately
        for entry in eval_data:
            client.log_feedback(
                dataset_name=dataset_name,
                run_name=f"RAG_Evaluation_{model_name}_{uuid4().hex[:8]}",
                feedback=entry,  # Send structured evaluation data
                tags=[model_name, "ragas"],  # Track different models
            )

        print("Successfully uploaded evaluation results to LangSmith.")

    except Exception as e:
        print(f"Error uploading to LangSmith: {e}")
        import traceback
        traceback.print_exc()  # This will give more detail about the error






# don't delete this 
def upload_to_langsmith_observability(evaluation_results, model_name):
    """Uploads the evaluation results to LangSmith for comparison across models."""
    try:
        client = Client()  # Ensure the LangSmith client is initialized
        run_name = f"RAG_Evaluation_{model_name}_{uuid4().hex[:8]}"  # Unique per model

        # Check the type and content of evaluation_results
        print("Type of evaluation_results:", type(evaluation_results))
        print("Contents of evaluation_results:", evaluation_results)

        # Ensure evaluation_results has the expected structure
        if not hasattr(evaluation_results, "scores"):
            raise ValueError("evaluation_results does not contain 'scores' attribute")

        # Extract and sanitize scores
        metrics_dict = {
            k: float(v) if isinstance(v, (np.floating, float)) else (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in evaluation_results.scores.items()
        }

        print("Processed metrics_dict:", metrics_dict)

        # Create a project if it doesn't exist
        project_name = "RAG_Evaluations"
        try:
            client.create_project(project_name=project_name)
        except Exception:
            pass  # Project likely already exists, no need to log error

        # Ensure 'question' key exists in the results before using it
        if isinstance(evaluation_results, dict) and "question" not in evaluation_results:
            print("Warning: 'question' key missing in evaluation_results")
            evaluation_results["question"] = "Unknown Question"

        # Create the run with a valid run_type value
        run = client.create_run(
            run_type="chain",  # Using "chain" as the run_type
            project_name=project_name,
            name=run_name,
            inputs={"model_name": model_name, "question": evaluation_results.get("question", "Unknown Question")},
            outputs={"metrics": metrics_dict},
            tags=[model_name, "ragas"]  # You can add EMBEDDINGS_MODEL if defined
        )

        print(f"Successfully uploaded evaluation results to LangSmith: {run_name}")
        print(f"Run ID: {run.id if hasattr(run, 'id') else 'N/A'}")

    except Exception as e:
        print(f"Error uploading to LangSmith: {e}")
        import traceback
        traceback.print_exc()  # This will give more detail about the error












def main():
    csv_path = "evaluation_results.csv"
    #run_langsmith_evaluation(csv_path)


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
    # Example usage in `main()`:
    

    print(f"LangSmith API Key: {os.getenv('LANGCHAIN_API_KEY')[:5]}...")
    print(f"LangSmith Endpoint: {os.getenv('LANGCHAIN_ENDPOINT')}")
    print("Evaluation results structure:")
    print(type(evaluation_results))


    print('---------------------')
    print(evaluation_results)
    print('---------------------')
    #upload_to_langsmith_evaluation(evaluation_results, MODEL)  # Pass the model name
    # Example Usage
  
    save_ragas_results_to_csv(evaluation_results, csv_path)
    #upload_csv_to_langsmith(csv_path)
    


    print("Done.")

if __name__ == "__main__":
    main()