class SQLQueries:
    """Static class to store all SQL queries."""

    CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS document_store_ext (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_name TEXT NOT NULL,
        document_id TEXT NOT NULL UNIQUE,        
        document_title TEXT NOT NULL,
        document_title_hash TEXT NOT NULL UNIQUE,
        subject TEXT NOT NULL,
        ttl_days INTEGER NOT NULL DEFAULT 365,
        document_meta_data TEXT NOT NULL,
        load_date DATETIME NOT NULL DEFAULT (datetime('now', '-7 hours'))
    );
    """

    INSERT_DOCUMENT = """
    INSERT INTO document_store_ext (document_id, document_name, document_title, document_title_hash, ttl_days, document_meta_data, subject)
    VALUES (?, ?, ?, ?, ?, ?, ?);
    """

    GET_DOCUMENT_BY_HASH = """
    SELECT * FROM document_store_ext WHERE document_title_hash = ?;
    """

    GET_ALL_DOCUMENTS = """
    SELECT * FROM document_store_ext;
    """

    DECREMENT_TTL = """
    UPDATE document_store_ext 
    SET ttl_days = MAX(365 - (strftime('%s', 'now') - strftime('%s', load_date)) / 86400, 0);
    """

    DELETE_EXPIRED = """
    DELETE FROM document_store_ext WHERE ttl_days <= 0;
    """
