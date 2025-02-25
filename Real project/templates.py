from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import List, Optional
from logger import logger

class MetaDataModel(BaseModel):
    """Standard metadata model with required and optional fields."""
    
    # REQUIRED fields
    document_name: str = Field(..., description="File name of the document")
    document_id: str = Field(..., description="Unique document identifier")
    document_date: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        description="Date of the document creation"
    )
    title: str = Field(..., min_length=1, max_length=255, description="Title of the document")
    chunk_number: int = Field(default=1, description="Used to put the document back together from the vector store")

    # OPTIONAL fields
    description: Optional[str] = Field(default="No description provided.", description="Brief document description")
    author: Optional[str] = Field(default="Anonymous", description="Author of the document")
    tags: Optional[List[str]] = Field(default_factory=lambda: ["tag1", "tag2", "tag3"], description="Tags for categorization")
    subject: Optional[str] = Field(default="General", description="Subject of the document")

    class Config:
        orm_mode = True  # Enables compatibility with ORMs like SQLAlchemy

    def to_dict(self):
        """Convert the model to a dictionary."""
        return self.model_dump()   

    @classmethod
    def from_dict(cls, metadata_dict: dict) -> "MetaDataModel":
        """Creates a MetaDataModel instance from a dictionary."""
        return cls(**metadata_dict)


if __name__ == "__main__":
    print("Ready Player 1")
    # Test the MetaDataModel
    metadata = MetaDataModel(
        document_id="12345",
        title="My First Document"
    )
    logger.debug(metadata.model_dump_json(indent=4))    
    logger.debug("Done")