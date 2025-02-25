import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint, Filter, FieldCondition, MatchValue, PointStruct, Distance, VectorParams, MatchValue
from utils_openai import UtilityOpenAI
from templates import MetaDataModel
from logger import logger
from typing import List, Optional
from dotenv import load_dotenv
import os
import hashlib

load_dotenv()

class UtilityQdrant:
    def __init__(self, collection_name: str, embedding_dim: int = 1536, hit_score: float = 0.60):
        self.local_store = False
        self.api_key = os.getenv("QDRANT-CLOUD-API-KEY")

        if self.local_store:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient("https://40c458f2-24a9-4153-b15b-0addf6a6bbcf.us-east-1-0.aws.cloud.qdrant.io:6333", api_key=self.api_key)  
            

        self.COLLECTION_NAME = collection_name  # "qt_document_collection"
        self.create_collection(self.COLLECTION_NAME, embedding_dim)
        self.hit_score = hit_score # used for tuning the search results

    def create_collection(self, collection_name, embedding_dim):
        """Creates a collection in Qdrant."""
        # Check if the collection exists
        collections = self.client.get_collections().collections  # Extract collection objects
        collection_names = [col.name for col in collections]  # Extract collection names

        if collection_name in collection_names:
            logger.info(f"Collection '{collection_name}' already exists")
        else:
            # Create the collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
            )            
            logger.info(f"Collection '{collection_name}' created")

    def show_all_document_metadata(self):
        """Displays all metadata for documents in the collection."""
        scroll_result, _ = self.client.scroll(collection_name=self.COLLECTION_NAME) 
        for doc in scroll_result:  # Iterate through the list of documents
            logger.info(doc.payload)  # Now doc has a `.payload` attribute


    def delete_document(self, document_id: str):
        """Deletes all vectors associated with a given document_id."""
        
        # Create a valid filter using Qdrant's `Filter` model
        delete_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )
            ]
        )

        # Perform the deletion with the correct filter format
        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=delete_filter  # Use the filter to select the points to delete
        )

        logger.info(f"Deleted all vectors for document_id: {document_id}")


    def insert_documents(self, collection_name, vectored_data, vectored_metadata):
        """Inserts chunked documents into a Qdrant collection with metadata."""

        logger.debug(f"Inserted metadata payload: {vectored_metadata}")


        # Ensure vectored_metadata is a dictionary (not a list)
        if not isinstance(vectored_metadata, dict):
            raise ValueError("Metadata must be a dictionary.")

        # Ensure vectored_data is a list of embeddings
        if not isinstance(vectored_data, list) or not all(isinstance(v, list) for v in vectored_data):
            raise ValueError("vectored_data must be a list of vector lists.")
        
        logger.debug(f"Embedding Count: {len(vectored_data)}")

        logger.debug("Metadata:", vectored_metadata)

        # Create the points
        points = [
            PointStruct(
                id=hashlib.md5(str(vector).encode()).hexdigest(),  # Hash each individual vector
                vector=vector,
                payload=vectored_metadata
            )
            for vector in vectored_data
            ]

        # Insert into Qdrant
        self.client.upsert(collection_name=collection_name, points=points)
        logger.debug(f"Inserted {len(points)} documents into '{collection_name}'")

    
    def search(self, collection_name, query_vector, top_k=3)-> list:
        """Search for documents in Qdrant using an embedding vector of the query."""
        hits  = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k
        ).points
        logger.debug(f"Found {len(hits)} hits")   
        # Filter hits based on score
        return_hits = []
        for hit in hits:
            if hit.score > self.hit_score:
                return_hits.append({
                    "score": hit.score,
                    "metadata": hit.payload
                })

        if not return_hits:
            logger.debug("No relevant documents found.")
            return_hits.append({
                "score": 0,
                "metadata": "No relevant documents found."
            })

        return return_hits
        # extras for prototyping
        # query_filter=models.Filter(
        #must=[models.FieldCondition(key="year", range=models.Range(gte=2000))])
        #for hit in hits:
         #   logger.debug(hit.payload, "score:", hit.score)



        


if __name__ == '__main__':
    print("Ready player one")
    COLLECTION_NAME = "qt_document_collection"
    
    # Initialize OpenAI Utility
    utility = UtilityOpenAI()
    embedding_dim = utility.get_embedding_dimension()
    
    # Initialize Qdrant
    qdrant = UtilityQdrant(COLLECTION_NAME, embedding_dim)    
    
    # Sample text chunks
    test_chunks = [
        "This is the first test chunk.",
        "This is the second test chunk.",
        "This is the third test chunk."
    ]

    # Generate embeddings
    vectors = utility.create_embeddings_from_text(test_chunks)

    # Sample metadata for each document
    #OLD WOKRING metadata_list = [
    #    {"document_id": "doc1", "title": "First Document", "author": "Alice"},
    #    {"document_id": "doc2", "title": "Second Document", "author": "Bob"},
    #    {"document_id": "doc3", "title": "Third Document", "author": "Charlie"},
    #    {"document_id": "doc4", "title": "Machine Learning", "author": "John"},
    #]

    metadata_list = [
        MetaDataModel(document_name = "test1", document_id="doc1", title="First Document", author="Alice", chunk_number=1),
        MetaDataModel(document_name = "test2", document_id="doc2", title="Second Document", author="Bob", chunk_number=1),
        MetaDataModel(document_name = "test3", document_id="doc3", title="Third Document", author="Charlie", chunk_number=1),
        MetaDataModel(document_name = "test4", document_id="doc4", title="Machine Learning", author="John", chunk_number=1),
    ]




    #metadata = []

    # Insert embeddings with metadata
    for i in range(3):        
        qdrant.insert_documents(COLLECTION_NAME, [vectors[i]], metadata_list[i].to_dict())

    #qdrant.insert_documents(COLLECTION_NAME, vectors, metadata_list)




    documents = [
    "In machine learning, feature scaling is the process of normalizing the range of independent variables or features. The goal is to ensure that all features contribute equally to the model, especially in algorithms like SVM or k-nearest neighbors where distance calculations matter.",
   
    "Feature scaling is commonly used in data preprocessing to ensure that features are on the same scale. This is particularly important for gradient descent-based algorithms where features with larger scales could disproportionately impact the cost function.",
   
    "In data science, feature extraction is the process of transforming raw data into a set of engineered features that can be used in predictive models. Feature scaling is related but focuses on adjusting the values of these features.",
   
    "Unsupervised learning algorithms, such as clustering methods, may benefit from feature scaling as it ensures that features with larger numerical ranges don't dominate the learning process.",
   
    "One common data preprocessing technique in data science is feature selection. Unlike feature scaling, feature selection aims to reduce the number of input variables used in a model to avoid overfitting.",
   
    "Principal component analysis (PCA) is a dimensionality reduction technique used in data science to reduce the number of variables. PCA works best when data is scaled, as it relies on variance which can be skewed by features on different scales.",
   
    "Min-max scaling is a common feature scaling technique that usually transforms features to a fixed range [0, 1]. This method is useful when the distribution of data is not Gaussian.",
   
    "Standardization, or z-score normalization, is another technique that transforms features into a mean of 0 and a standard deviation of 1. This method is effective for data that follows a normal distribution.",
   
    "Feature scaling is critical when using algorithms that rely on distances, such as k-means clustering, as unscaled features can lead to misleading results.",
   
    "Scaling can improve the convergence speed of gradient descent algorithms by preventing issues with different feature scales affecting the cost function's landscape.",
   
    "In deep learning, feature scaling helps in stabilizing the learning process, allowing for better performance and faster convergence during training.",
   
    "Robust scaling is another method that uses the median and the interquartile range to scale features, making it less sensitive to outliers.",
   
    "When working with time series data, feature scaling can help in standardizing the input data, improving model performance across different periods.",
   
    "Normalization is often used in image processing to scale pixel values to a range that enhances model performance in computer vision tasks.",
   
    "Feature scaling is significant when features have different units of measurement, such as height in centimeters and weight in kilograms.",
   
    "In recommendation systems, scaling features such as user ratings can improve the model's ability to find similar users or items.",
   
    "Dimensionality reduction techniques, like t-SNE and UMAP, often require feature scaling to visualize high-dimensional data in lower dimensions effectively.",
   
    "Outlier detection techniques can also benefit from feature scaling, as they can be influenced by unscaled features that have extreme values.",
   
    "Data preprocessing steps, including feature scaling, can significantly impact the performance of machine learning models, making it a crucial part of the modeling pipeline.",
   
    "In ensemble methods, like random forests, feature scaling is not strictly necessary, but it can still enhance interpretability and comparison of feature importance.",
   
    "Feature scaling should be applied consistently across training and test datasets to avoid data leakage and ensure reliable model evaluation.",
   
    "In natural language processing (NLP), scaling can be useful when working with numerical features derived from text data, such as word counts or term frequencies.",
   
    "Log transformation is a technique that can be applied to skewed data to stabilize variance and make the data more suitable for scaling.",
   
    "Data augmentation techniques in machine learning may also include scaling to ensure consistency across training datasets, especially in computer vision tasks."
    ]

    vectors = utility.create_embeddings_from_text(test_chunks)
    metadata = MetaDataModel(document_name="test6", document_id="doc5", title="Document number 5 Chunk", author="Seraphina", chunk_number=1)
    qdrant.insert_documents(COLLECTION_NAME, vectors, metadata.to_dict())

    logger.debug("Embeddings and metadata inserted successfully.")

    query = "Find like first chunk"
    # Ensure the query is a list for compatibility
    query_vector = utility.create_embeddings_from_text([query])[0]  
    search_results = qdrant.search(COLLECTION_NAME, query_vector, 3)

    # Print results
    logger.debug("--------------------")
    logger.debug("\n🔍 Search Results:")
    for result in search_results:
        logger.debug(f"Score: {result['score']}, Metadata: {result['metadata']}")



    logger.debug("--------------------")

    qdrant.show_all_document_metadata()        
    qdrant.delete_document("doc2")
    qdrant.show_all_document_metadata()






    #query = "What is the purpose of feature scaling in machine learning?"
    #vector_query = utility.create_embeddings_from_text(query)
    #search_results = qdrant.search(COLLECTION_NAME, vector_query, utility)
    ## Print results
    #print("\n🔍 Search Results:")
    #for result in search_results:
    #    print(f"Score: {result['score']}, Metadata: {result['metadata']}")

    
