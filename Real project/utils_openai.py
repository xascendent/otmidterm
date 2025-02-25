import os
from enum import Enum
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from logger import logger

load_dotenv()

class GptEmbeddingModel(Enum):
    SMALL3 = "text-embedding-3-small"
    LARGE3 = "text-embedding-3-large"

class UtilityOpenAI:
    def __init__(self, api_key: str = None, model: GptEmbeddingModel = GptEmbeddingModel.SMALL3):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required.")    
        self.embedding = OpenAIEmbeddings(api_key=api_key, model=model)
        

    def create_embeddings_from_text(self, chunks: list[str]) -> list[str]:
        # check to see if the list is empty
        if not chunks:
            raise ValueError("List of text chunks cannot be empty.")
        logger.debug("Chunks to embed:", chunks)
        vectors = self.embedding.embed_documents(chunks)    
        logger.debug(f"Number of vectors: {len(vectors)}")
        logger.debug(f"First vector: {vectors[0]}")
        logger.debug(f"Last vector: {vectors[-1]}")
        return vectors
    
    def get_embedding_dimension(self) -> int:
        """Retrieve the dimensionality of the embedding model."""
        embedding_dim = len(self.embedding.embed_query("test"))
        return embedding_dim
        
        
if __name__ == "__main__":
    # Test code or example usage
    test_chunks = [
        "This is the first test chunk.",
        "This is the second test chunk.",
        "This is the third test chunk."
    ]

    utility = UtilityOpenAI()
    vectors = utility.create_embeddings_from_text(test_chunks)