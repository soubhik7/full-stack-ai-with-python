from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from typing import List, Dict, Any
from src.gmail_assistant.core.assistant import GmailAIAssistant
from src.gmail_assistant.api.schemas import SummaryItem, ScanTriggerResponse
from src.gmail_assistant.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Global assistant instance to be initialized at startup
assistant: Optional[GmailAIAssistant] = None

# Store latest summaries in memory for the web interface
latest_summaries: List[Dict[str, str]] = []

def init_assistant(assistant_instance: GmailAIAssistant):
    """Initialize the global assistant instance."""
    global assistant
    assistant = assistant_instance

@router.get("/", response_class=HTMLResponse)
async def read_root():
    """Returns the latest summaries in a professional web interface."""
    if not latest_summaries:
        return """
        <html>
            <head>
                <title>Gmail AI Assistant</title>
                <style>
                    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f5; color: #1c1e21; text-align: center; padding-top: 50px; }
                    .card { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 40px; display: inline-block; max-width: 500px; }
                    h1 { color: #1877f2; }
                    .btn { background-color: #1877f2; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; text-decoration: none; font-weight: bold; }
                    .btn:hover { background-color: #166fe5; }
                </style>
            </head>
            <body>
                <div class="card">
                    <h1>Gmail AI Assistant</h1>
                    <p>No email summaries have been generated yet.</p>
                    <a href="/trigger" class="btn">Scan for New Emails</a>
                </div>
            </body>
        </html>
        """
    
    # Use the processor's HTML generation logic
    from src.gmail_assistant.core.processor import EmailProcessor
    # We need a temporary processor instance for HTML generation
    processor = EmailProcessor(None, None) 
    html_content = processor.generate_summary_brief_html(latest_summaries)
    
    # Add a 'Scan Again' button to the generated HTML
    scan_again_btn = '<div style="text-align: center; padding: 20px;"><a href="/trigger" style="background-color: #667eea; color: white; padding: 10px 20px; text-decoration: none; border-radius: 6px; font-weight: bold;">Scan for New Emails</a></div>'
    html_content = html_content.replace('<div class="content">', scan_again_btn + '<div class="content">')
    
    return html_content

@router.get("/trigger", response_model=ScanTriggerResponse)
async def trigger_scan(background_tasks: BackgroundTasks):
    """
    Triggers an email scan and summarization process.
    Updates the global list of latest summaries.
    """
    global latest_summaries
    if assistant is None:
        raise HTTPException(status_code=500, detail="Assistant not properly initialized.")
    
    try:
        # Run synchronously for immediate feedback in this simple version
        # but in a high-load app, background_tasks.add_task would be used.
        summaries = assistant.run_scan_and_notify(max_emails=5)
        latest_summaries = summaries
        
        return {
            "status": "success", 
            "processed_count": len(summaries), 
            "summaries": summaries
        }
    except Exception as e:
        logger.error(f"Error during scan trigger: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process emails: {str(e)}")

@router.get("/api/summaries", response_model=List[SummaryItem])
async def get_summaries():
    """Returns the latest processed summaries in JSON format."""
    return latest_summaries
