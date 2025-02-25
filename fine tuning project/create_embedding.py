import json
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import wandb
import os
from dotenv import load_dotenv
from huggingface_hub import login  



load_dotenv()


access_token = os.getenv("HUGGING_FACE_TOKEN")

login(token=access_token)



# Constants
BATCH_SIZE = 10
EPOCHS = 10
MODEL_ID = "Snowflake/snowflake-arctic-embed-l"
HF_USERNAME = "shivXy"
OUTPUT_DIR = "finetuned_arctic_ft"

# Load JSONL dataset
def load_jsonl(file_path):
    data = {"corpus": {}, "questions": {}, "relevant_contexts": {}}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if "corpus" in entry:
                data["corpus"].update(entry["corpus"])
            if "questions" in entry:
                data["questions"].update(entry["questions"])
            if "relevant_contexts" in entry:
                data["relevant_contexts"].update(entry["relevant_contexts"])
    return data

# Load training and validation datasets
train_dataset = load_jsonl("training_dataset.jsonl")
val_dataset = load_jsonl("val_dataset.jsonl")

# Initialize SentenceTransformer model
model = SentenceTransformer(MODEL_ID)
wandb.init(mode="disabled")

# Prepare training examples
examples = []
corpus = train_dataset["corpus"]
queries = train_dataset["questions"]
relevant_docs = train_dataset["relevant_contexts"]

for query_id, query in queries.items():
    if query_id in relevant_docs and relevant_docs[query_id]:  # Ensure key exists
        doc_id = relevant_docs[query_id][0]
        if doc_id in corpus:  # Ensure doc_id exists in corpus
            text = corpus[doc_id]
            examples.append(InputExample(texts=[query, text]))

# Create DataLoader
loader = DataLoader(examples, batch_size=BATCH_SIZE, shuffle=True)

# Prepare evaluator
corpus = val_dataset["corpus"]
queries = val_dataset["questions"]
relevant_docs = val_dataset["relevant_contexts"]
evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs)

# Define loss functions
matryoshka_dimensions = [768, 512, 256, 128, 64]
inner_train_loss = MultipleNegativesRankingLoss(model)
train_loss = MatryoshkaLoss(model, inner_train_loss, matryoshka_dims=matryoshka_dimensions)

# Calculate warmup steps
warmup_steps = int(len(loader) * EPOCHS * 0.1)

# Train model
model.fit(
    train_objectives=[(loader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    output_path=OUTPUT_DIR,
    show_progress_bar=True,
    evaluator=evaluator,
    evaluation_steps=50,
)

# Push model to Hugging Face Hub
model.push_to_hub(f"{HF_USERNAME}/ot-midterm-v0")
