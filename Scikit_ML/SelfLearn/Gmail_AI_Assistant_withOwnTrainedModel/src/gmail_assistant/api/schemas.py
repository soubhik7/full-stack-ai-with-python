from pydantic import BaseModel, Field
from typing import List, Optional

class SummaryItem(BaseModel):
    """
    Pydantic schema for a single email summary item.
    """
    id: str = Field(..., description="The unique ID of the Gmail message.")
    sender: str = Field(..., description="The sender's name or email address.")
    subject: str = Field(..., description="The subject of the email.")
    summary: str = Field(..., description="The AI-generated extractive summary.")

class ScanTriggerResponse(BaseModel):
    """
    Pydantic schema for the response after triggering an email scan.
    """
    status: str = Field(..., description="The status of the scan operation.")
    processed_count: int = Field(..., description="The number of emails successfully processed.")
    summaries: List[SummaryItem] = Field(..., description="A list of generated email summaries.")

class ErrorResponse(BaseModel):
    """
    Pydantic schema for error responses.
    """
    detail: str = Field(..., description="A detailed error message.")
