from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os

from app import process_emails, generate_html_brief

app = FastAPI(title="Gmail AI Assistant API")

# Store summaries in memory for simplicity (in a real app, use a database)
latest_summaries = []

class SummaryItem(BaseModel):
    id: str
    sender: str
    subject: str
    summary: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Returns the latest summaries in a web interface."""
    if not latest_summaries:
        return """
        <html>
            <head><title>Gmail AI Assistant</title></head>
            <body>
                <h1>Gmail AI Assistant</h1>
                <p>No summaries yet. <a href="/trigger">Trigger a scan now</a>.</p>
            </body>
        </html>
        """
    
    html_content = generate_html_brief(latest_summaries)
    # Add a link to trigger again
    html_content = html_content.replace("Processed", f'<a href="/trigger">Scan Again</a><br>Processed')
    return html_content

@app.get("/trigger")
async def trigger_scan(background_tasks: BackgroundTasks):
    """Triggers an email scan and summarization in the background."""
    global latest_summaries
    try:
        # For simplicity, we run it synchronously here to get immediate results for the user
        # but in a production app, background_tasks would be better.
        latest_summaries = process_emails(max_emails=5)
        return {"status": "success", "processed_count": len(latest_summaries), "summaries": latest_summaries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/summaries", response_model=List[SummaryItem])
async def get_summaries():
    """Returns the latest processed summaries as JSON."""
    return latest_summaries

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
