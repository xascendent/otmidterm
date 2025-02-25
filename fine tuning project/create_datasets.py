import nest_asyncio
import os
import asyncio
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tqdm.asyncio import tqdm
import json

load_dotenv()
nest_asyncio.apply()

api_key = os.getenv("OPENAI_API_KEY")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=20,
    length_function=len
)

current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_dir = os.path.join(current_dir, "data/")
text_loader = DirectoryLoader(pdf_dir, glob="*.pdf", loader_cls=PyPDFLoader)

training_documents = text_splitter.split_documents(text_loader.load())

print(f"Total Documents: {len(training_documents)}")

id_set = set()
for document in training_documents:
    doc_id = str(uuid.uuid4())
    while doc_id in id_set:
        doc_id = str(uuid.uuid4())
    id_set.add(doc_id)
    document.metadata["id"] = doc_id

training_split_documents = training_documents[:len(training_documents) - 24]  # take chunks 78
val_split_documents = training_documents[78:89]  # 78 to 89
test_split_documents = training_documents[102 - 12:]  # 90 to 101 chunks

print(val_split_documents)

qa_chat_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

qa_prompt = """\
Given the following context, you must generate questions based on only the provided context.

You are to generate {n_questions} questions which should be provided in the following format:

1. QUESTION #1
2. QUESTION #2
...

Context:
{context}
"""

qa_prompt_template = ChatPromptTemplate.from_template(qa_prompt)
question_generation_chain = qa_prompt_template | qa_chat_model


async def create_questions(documents, n_questions):
    questions = {}
    relevant_docs = {}

    for doc in tqdm(documents, desc="Generating Questions"):
        context = doc.page_content
        doc_id = doc.metadata["id"]

        try:
            # Generate questions asynchronously
            response = await question_generation_chain.ainvoke({"context": context, "n_questions": n_questions})
            generated_questions = response.content.strip().split("\n")

            for q in generated_questions:
                if q.strip():
                    question_id = str(uuid.uuid4())
                    questions[question_id] = q.strip()
                    relevant_docs[question_id] = [doc_id]
        except Exception as e:
            print(f"Error generating questions for doc {doc_id}: {e}")

    return questions, relevant_docs


async def main():
    run = False

    if run:
        # Generate training set
        training_questions, training_relevant_contexts = await create_questions(training_split_documents, 2)
        val_questions, val_relevant_contexts = await create_questions(val_split_documents, 2)
        test_questions, test_relevant_contexts = await create_questions(test_split_documents, 2)

        # Training dataset
        training_corpus = {train_item.metadata["id"]: train_item.page_content for train_item in training_split_documents}
        train_dataset = {
            "questions": training_questions,
           "relevant_contexts": training_relevant_contexts,
            "corpus": training_corpus
        }

        with open("training_dataset.jsonl", "w") as f:
            json.dump(train_dataset, f)

        # Validation dataset
        val_corpus = {val_item.metadata["id"]: val_item.page_content for val_item in val_split_documents}
        val_dataset = {
            "questions": val_questions,
            "relevant_contexts": val_relevant_contexts,
            "corpus": val_corpus
        }

        with open("val_dataset.jsonl", "w") as f:
            json.dump(val_dataset, f)

        # Test dataset
        test_corpus = {test_item.metadata["id"]: test_item.page_content for test_item in test_split_documents}
        test_dataset = {
            "questions": test_questions,
            "relevant_contexts": test_relevant_contexts,
            "corpus": test_corpus
        }

        with open("test_dataset.jsonl", "w") as f:           
           json.dump(test_dataset, f)

    


# Run the async function
asyncio.run(main())
