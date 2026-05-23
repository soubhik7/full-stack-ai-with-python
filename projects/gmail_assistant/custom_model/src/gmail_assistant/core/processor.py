from typing import List, Dict, Any, Optional
from src.gmail_assistant.services.gmail_service import GmailService
from src.gmail_assistant.ml.inference import Summarizer
from src.gmail_assistant.utils.logger import get_logger

logger = get_logger(__name__)

class EmailProcessor:
    """
    Orchestrates the process of fetching emails, generating summaries, and marking them as read.
    It acts as the bridge between the Gmail service and the ML summarization model.
    """
    def __init__(self, gmail_service: GmailService, summarizer: Summarizer):
        """
        Initialize the EmailProcessor with required services.
        
        Args:
            gmail_service: Initialized GmailService for interacting with Gmail API.
            summarizer: Initialized Summarizer for generating extractive summaries.
        """
        self.gmail_service = gmail_service
        self.summarizer = summarizer

    def process_latest_unread(self, max_emails: int = 5) -> List[Dict[str, str]]:
        """
        Fetches the latest unread emails, summarizes each, and returns the list of summaries.
        
        Args:
            max_emails: Maximum number of unread emails to process.
            
        Returns:
            A list of dictionaries, each containing email metadata and its summary.
        """
        logger.info(f"Processing up to {max_emails} latest unread emails...")
        
        messages = self.gmail_service.get_unread_messages(max_results=max_emails)
        if not messages:
            logger.info("No unread messages found.")
            return []
            
        summaries = []
        for msg in messages:
            msg_id = msg['id']
            try:
                # 1. Fetch full message content
                content = self.gmail_service.get_message_content(msg_id)

                if self.gmail_service.is_subject_excluded(content.get('subject', '')):
                    self.gmail_service.mark_as_read(msg_id)
                    logger.info(f"Skipped excluded subject for message {msg_id}.")
                    continue
                
                # 2. Generate summary using the ML model
                summary = self.summarizer.summarize(content['body'], top_n=2)
                
                # 3. Handle cases where summary might be empty
                if not summary:
                    summary = content['body'][:200] + "..."
                
                summaries.append({
                    'id': msg_id,
                    'sender': content['sender'],
                    'subject': content['subject'],
                    'summary': summary.replace('\n', ' ')
                })
                
                # 4. Mark message as read
                self.gmail_service.mark_as_read(msg_id)
                logger.info(f"Successfully processed message {msg_id}.")
                
            except Exception as e:
                logger.error(f"Error processing message {msg_id}: {e}")
                continue
                
        return summaries

    def generate_summary_brief_html(self, summaries: List[Dict[str, str]]) -> str:
        """
        Generates a professionally formatted HTML brief from the list of summaries.
        """
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
          <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Your AI Email Brief</title>
            <style>
              body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background-color: #f4f7f6; margin: 0; padding: 20px 0; color: #333333; }
              .container { max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
              .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px 20px; text-align: center; color: #ffffff; }
              .header h1 { margin: 0; font-size: 24px; font-weight: 600; letter-spacing: 0.5px; }
              .content { padding: 20px 25px; }
              .email-card { background-color: #ffffff; border: 1px solid #e1e5eb; border-left: 4px solid #667eea; border-radius: 6px; padding: 16px; margin-bottom: 20px; }
              .email-sender { font-size: 13px; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
              .email-subject { font-size: 16px; font-weight: bold; color: #1f2937; margin: 5px 0; line-height: 1.4; }
              .email-summary { font-size: 14px; color: #4b5563; line-height: 1.6; margin-top: 10px; }
              .footer { background-color: #f9fafb; padding: 20px; text-align: center; font-size: 12px; color: #9ca3af; border-top: 1px solid #e5e7eb; }
            </style>
          </head>
          <body>
            <div class="container">
              <div class="header">
                <h1>📥 AI Email Brief</h1>
                <p>Your latest unread emails, condensed into a quick reading list.</p>
              </div>
              <div class="content">
        """
        
        for item in summaries:
            html_template += f"""
                <div class="email-card">
                  <div class="email-sender">{item['sender']}</div>
                  <h3 class="email-subject">{item['subject']}</h3>
                  <div class="email-summary">{item['summary']}</div>
                </div>
            """
            
        html_template += f"""
              </div>
              <div class="footer">
                Generated automatically by your Personal AI Assistant.<br>
                Processed {len(summaries)} new items.
              </div>
            </div>
          </body>
        </html>
        """
        return html_template
