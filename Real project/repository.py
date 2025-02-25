import uuid
import hashlib
from database import Database
from queries import SQLQueries
from logger import logger

class DocumentRepository:
    """Handles database operations for document store."""
    
    def __init__(self):
        self.db = Database()  # Singleton instance

        # Ensure table exists
        self.db.cursor.execute(SQLQueries.CREATE_TABLE)
        self.db.commit()

    def hash_title(self, title: str) -> str:
        """Generates SHA-256 hash for a document title."""
        cleaned_title = title.strip().upper()
        return hashlib.sha256(cleaned_title.encode()).hexdigest()

    def insert_document(self, document_id, document_name, title, metadata, subject, ttl_days=365):
        """Insert a document if the hashed title is unique."""
        title_hash = self.hash_title(title)
        # document_id = str(uuid.uuid4())

        # Check if document already exists
        self.db.cursor.execute(SQLQueries.GET_DOCUMENT_BY_HASH, (title_hash,))
        exists = self.db.cursor.fetchone()

        if exists:
            logger.info(f"Document '{title}' already exists. Skipping insert.")
            return False  # Already exists

        # Insert new document
        self.db.cursor.execute(SQLQueries.INSERT_DOCUMENT, (document_id, document_name, title, title_hash, ttl_days, metadata, subject))
        self.db.commit()
        logger.info(f"Inserted document: {title}")
        return True  # Successfully inserted

    def decrement_ttl(self):
        """Decrements ttl_days for all records."""
        self.db.cursor.execute(SQLQueries.DECREMENT_TTL)
        self.db.commit()
        logger.info("TTL decremented for all documents.")

    def delete_expired_documents(self):
        """Deletes documents where ttl_days <= 0."""
        self.db.cursor.execute(SQLQueries.DELETE_EXPIRED)
        deleted_count = self.db.cursor.rowcount
        self.db.commit()
        logger.info(f"Deleted {deleted_count} expired documents.")

    def get_all_documents(self):
        """Retrieve all documents from the store."""
        self.db.cursor.execute(SQLQueries.GET_ALL_DOCUMENTS)
        documents = self.db.cursor.fetchall()
        return documents

if __name__ == "__main__":
    print("Ready Player 1")
    repo = DocumentRepository()    
    repo.insert_document(str(uuid.uuid4()), "ready1.pdf", "Ready Player 1", "A book about virtual reality and gaming.", "Science Fiction")
    repo.insert_document(str(uuid.uuid4()), "ready2.pdf", "Ready Player 2", "A sequel to the first book.", "Science Fiction")
    repo.decrement_ttl()
    repo.delete_expired_documents()
    documents = repo.get_all_documents()
    for doc in documents:
        logger.debug(doc)
    logger.debug("Done")
