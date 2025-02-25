import sqlite3

class Database:
    """Singleton for managing SQLite connection."""
    
    _instance = None

    def __new__(cls, db_path="document_store.db"):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance.connection = sqlite3.connect(db_path, check_same_thread=False)
            cls._instance.cursor = cls._instance.connection.cursor()
        return cls._instance

    def commit(self):
        """Commit changes to the database."""
        self.connection.commit()

    def close(self):
        """Close the database connection."""
        self.connection.close()
