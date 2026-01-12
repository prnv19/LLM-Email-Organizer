import imaplib
import email
from email.policy import default
import os
import time
import select
from dotenv import load_dotenv

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser

# from langchain.chains import LLMChain
from pydantic import BaseModel
from typing import List

load_dotenv()

class EmailBinaryOnly(BaseModel):
    binary: List[int]  # Multi-label binary vector

class LLMClassifierDB:
    def __init__(self):
        llm = ChatGoogleGenerativeAI(
                    model="gemini-3-flash-preview",
                    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    api_key=os.getenv("GOOGLE_API_KEY"),
                    )
        
        parser = JsonOutputParser(pydantic_object=EmailBinaryOnly)

        #labels
        self.LABELS = [
                    {"name": "Hiring", "description": "Notifications about potential job opportunities, updates, and matches from various job platforms."},
                    {"name": "Security Alerts", "description": "Security alerts for my accounts and bank-related stuff for all categories."},
                    {"name": "Finance", "description": "Transactions, receipts, statements, and promotions from banks and related financial services — including billing notices, receipts, and payment-confirmation emails for purchases, reservations, or services charged to your bank account or card."},
                    {"name": "Promotional", "description": "Promotional emails across all categories. Don't include job related stuff"},
                ]

        # Build label description string for the prompt
        label_text = "\n".join([f"{i}. {lbl['name']}: {lbl['description']}" for i, lbl in enumerate(self.LABELS)])

        # Prompt template
        example_json = '{"binary": [1, 0, 1, 0]}'

        prompt = ChatPromptTemplate.from_messages([
                ("system", """
            You are an assistant that classifies emails into multiple categories.
            Use the following labels and descriptions to classify the email:

            {labels}

            Return only a JSON object with a single field:
            - "binary": a list of 0/1 integers corresponding to whether each label applies.
            The order of the binary list must match the order above.
            Do not include label names or any other text.

            Example:
            {example}
            """),
                ("user", "Subject: {subject}\nBody: {body}")
            ]).partial(
                labels=label_text,
                example=example_json
            )
        
        self.llm_chain = prompt.pipe(llm).pipe(parser)


        # MongoDB Atlas Setup
        self.client = MongoClient(os.getenv("MONGO_DB_URI"), tlsCAFile=certifi.where(), server_api=ServerApi('1'))
        self.db = self.client["email_processor_db"]
        self.emails_collection = self.db["processed_emails"]
        
        # Create an index on message_id for faster lookups
        self.emails_collection.create_index("message_id", unique=True)
        print("✅ Connected to MongoDB Atlas")

    def is_already_processed(self, message_id):
        """Check if the email message_id exists in Atlas"""
        if not message_id:
            return False
        return self.emails_collection.find_one({"message_id": message_id}) is not None

    def mark_processed_in_db(self, message_id, subject, llm_result):
        """Store the processed record in Atlas"""
        record = {
            "message_id": message_id,
            "subject": subject,
            "llm_output": llm_result,
            "processed_at": time.time()
        }
        self.emails_collection.insert_one(record)

    def process_with_llm(self, subject, body):
        result = self.llm_chain.invoke({"subject": subject, "body": body})
        return result['binary']
    

class GmailListener(LLMClassifierDB):
    def __init__(self):
        super().__init__()
        self.email = os.getenv("GMAIL_USER")
        self.password = os.getenv("GMAIL_PASS")
        self.mail = None
        
    def connect(self):
        self.mail = imaplib.IMAP4_SSL("imap.gmail.com")
        self.mail.login(self.email, self.password)
        self.mail.select("INBOX")
        return self.mail
    
    def get_unread(self):
        _, data = self.mail.search(None, "UNSEEN")
        return data[0].split()

    def get_email_details(self, uid):
            """Safely fetch subject and Message-ID"""
            _, data = self.mail.fetch(uid, "(BODY.PEEK[HEADER])")
            
            # Check if data exists and data[0] is a tuple (standard imaplib response)
            if not data or not isinstance(data[0], tuple):
                return {"subject": "(no subject)", "message_id": None}
                
            raw_header = data[0][1]
            msg = email.message_from_bytes(raw_header, policy=default)
            return {
                "subject": msg.get("subject", "(no subject)"),
                "message_id": msg.get("Message-ID", None)
            }

    def get_body(self, uid):
        """Safely fetch and parse body"""
        _, data = self.mail.fetch(uid, "(BODY.PEEK[TEXT])")
        
        # Ensure data exists and is in expected format
        if not data or not isinstance(data[0], tuple):
            return ""
            
        raw_body = data[0][1]
        try:
            msg = email.message_from_bytes(raw_body, policy=default)
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        return part.get_content()
                return ""
            else:
                return msg.get_content()
        except Exception:
            return ""

    def process_email(self, uid):
        try:
            details = self.get_email_details(uid)
            subject = details['subject']
            m_id = details['message_id']
            
            print(f"\n📧 New email: {subject} (ID: {m_id})")
            
            # 1. Check MongoDB
            if not self.is_already_processed(m_id):
                try:
                    body = self.get_body(uid)
                    
                    # 2. LLM processing
                    llm_result = self.process_with_llm(subject, body)
                    print(f"   🤖 LLM Output: {llm_result}")

                    for idx, label in enumerate([label["name"] for label in self.LABELS]):
                        if llm_result[idx] == 1:
                            # Quote label if it has spaces (Gmail IMAP requirement)
                            gmail_label = f'"{label}"' if " " in label else label
                            self.mail.store(uid, '+X-GM-LABELS', gmail_label)
                            print(f"      🏷️  Applied label: {label}")
                    
                    # 3. Save to MongoDB
                    self.mark_processed_in_db(m_id, subject, llm_result)
                    print("   ✅ Record saved to Atlas")

                    #Debug : Add label to processed email
                    # self.mail.store(uid, '+X-GM-LABELS', 'system/ai_processed')
                except Exception as e:
                    print(f"   ❌ Failed to process: {e}")
            else:
                print("   ⏭️  Already exists in Database")
        except Exception as e:
            print(f"   ⚠️ Error processing email: {e}")

    def idle_loop(self):
        print("📬 Gmail IDLE listener started (MongoDB tracking active)")
        # ... (rest of your idle_loop remains the same)
        while True:
            try:
                uids = self.get_unread()
                for uid in uids:
                    self.process_email(uid)
                
                tag = self.mail._new_tag().decode()
                self.mail.send(f'{tag} IDLE\r\n'.encode())
                response = self.mail.readline()
                # print("   💤 Entered IDLE mode, waiting for new emails...")
                timeout = 1740 
                ready = select.select([self.mail.socket()], [], [], timeout)
                
                if ready[0]:
                    response = self.mail.readline()
                    self.mail.send(b'DONE\r\n')
                    self.mail.readline()
                    time.sleep(0.5)
                else:
                    self.mail.send(b'DONE\r\n')
                    self.mail.readline()
            except Exception as e:
                print(f"⚠️  Error: {e}")
                time.sleep(5)
                self.mail = self.connect()

    def start(self):
        try:
            self.mail = self.connect()
            self.idle_loop()
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
            if self.mail: self.mail.logout()

if __name__ == "__main__":
    listener = GmailListener()
    listener.start()