import os
import pickle
import base64
import logging
from email.mime.text import MIMEText
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import time
import re
from io import BytesIO
from typing import Dict, List, Tuple, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from tqdm import tqdm

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "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"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('resume_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    logger.error(f"Error downloading NLTK data: {e}")

class ResumeAnalyzer:
    def __init__(self):
        try:
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            self.setup_chain()
            logger.info("ResumeAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ResumeAnalyzer: {e}")
            raise

    def setup_chain(self):
        # Create the analysis chain with improved prompt
        self.analysis_chain = (
            ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are an expert HR recruiter and ATS system. Your task is to analyze a resume against a job description and provide a detailed assessment.

PURPOSE:
- Evaluate how well the candidate's resume matches the job requirements
- Identify key strengths and areas for improvement
- Provide a numerical score based on the match
- Give actionable insights for the hiring team

IMPORTANT: You will receive a job description and a resume text. You MUST analyze these documents and provide a score and detailed analysis.

Analysis Criteria:
1. Technical Skills Match (40% weight)
   - Required skills present (e.g., Python, Django, Flask)
   - Years of experience in required technologies
   - Technical proficiency level demonstrated

2. Experience Relevance (30% weight)
   - Relevant work history matching job requirements
   - Project complexity and scope
   - Industry experience and domain knowledge

3. Education & Certifications (15% weight)
   - Relevant degrees and qualifications
   - Professional certifications
   - Continuous learning and development

4. Soft Skills & Cultural Fit (15% weight)
   - Communication skills demonstrated
   - Team collaboration experience
   - Leadership potential and initiative

You MUST provide your analysis in this exact format:
Score: [number between 0-100, where 100 is perfect match]
Technical Skills: [detailed analysis of technical skills match]
Experience: [detailed analysis of relevant experience]
Education: [detailed analysis of education and certifications]
Soft Skills: [detailed analysis of soft skills and cultural fit]
Key Strengths: [list of top 3-5 strengths that match job requirements]
Areas for Improvement: [list of 2-3 areas that could be improved]
Overall Assessment: [comprehensive evaluation with specific examples from resume]"""),
                HumanMessage(content="""Here is the job description and resume to analyze:

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Please analyze the resume against the job description and provide your assessment in the required format. Focus on specific examples from the resume that demonstrate the candidate's qualifications.""")
            ])
            | self.llm
            | StrOutputParser()
        )

    def analyze_resume(self, resume_text: str, job_description: str) -> Dict:
        """Analyze a resume against job description"""
        try:
            if not resume_text.strip():
                logger.error("Empty resume text provided")
                return {
                    "score": 0,
                    "analysis": "Error: Empty resume text"
                }

            if not job_description.strip():
                logger.error("Empty job description provided")
                return {
                    "score": 0,
                    "analysis": "Error: Empty job description"
                }

            logger.info(f"Analyzing resume against job description (JD length: {len(job_description)}, Resume length: {len(resume_text)})")
            
            # Create a single prompt with all the information
            prompt = f"""Here is the job description and resume to analyze:

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Please analyze the resume against the job description and provide your assessment in the required format. Focus on specific examples from the resume that demonstrate the candidate's qualifications."""

            # Run the analysis with the complete prompt
            analysis = self.llm.invoke([
                SystemMessage(content="""You are an expert HR recruiter and ATS system. Your task is to analyze a resume against a job description and provide a detailed assessment.

PURPOSE:
- Evaluate how well the candidate's resume matches the job requirements
- Identify key strengths and areas for improvement
- Provide a numerical score based on the match
- Give actionable insights for the hiring team

Analysis Criteria:
1. Technical Skills Match (40% weight)
   - Required skills present (e.g., Python, Django, Flask)
   - Years of experience in required technologies
   - Technical proficiency level demonstrated

2. Experience Relevance (30% weight)
   - Relevant work history matching job requirements
   - Project complexity and scope
   - Industry experience and domain knowledge

3. Education & Certifications (15% weight)
   - Relevant degrees and qualifications
   - Professional certifications
   - Continuous learning and development

4. Soft Skills & Cultural Fit (15% weight)
   - Communication skills demonstrated
   - Team collaboration experience
   - Leadership potential and initiative

You MUST provide your analysis in this exact format:
Score: [number between 0-100, where 100 is perfect match]
Technical Skills: [detailed analysis of technical skills match]
Experience: [detailed analysis of relevant experience]
Education: [detailed analysis of education and certifications]
Soft Skills: [detailed analysis of soft skills and cultural fit]
Key Strengths: [list of top 3-5 strengths that match job requirements]
Areas for Improvement: [list of 2-3 areas that could be improved]
Overall Assessment: [comprehensive evaluation with specific examples from resume]"""),
                HumanMessage(content=prompt)
            ])
            
            # Print raw AI output for debugging
            print("\n=== Raw AI Model Output ===")
            print(analysis.content)
            print("=" * 50)
            
            # Log the analysis result
            logger.info("Analysis completed successfully")
            logger.debug(f"Analysis result: {analysis.content[:200]}...")
            
            # Extract score from analysis
            score_match = re.search(r'Score:\s*(\d+)', analysis.content)
            score = int(score_match.group(1)) if score_match else 0
            
            # Log the extracted score
            logger.info(f"Extracted score: {score}")
            
            return {
                "score": score,
                "analysis": analysis.content
            }
        except Exception as e:
            logger.error(f"Error analyzing resume: {e}")
            return {
                "score": 0,
                "analysis": f"Error during analysis: {str(e)}"
            }

class GmailResumeProcessor:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
        self.service = None
        self.credentials = None
        self.labels = None
        self.analyzer = ResumeAnalyzer()
        logger.info("GmailResumeProcessor initialized")

    def authenticate(self):
        """Authenticate with Gmail API"""
        try:
            creds = None
            if os.path.exists('token.pickle'):
                with open('token.pickle', 'rb') as token:
                    creds = pickle.load(token)
            
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials.json', self.SCOPES)
                    creds = flow.run_local_server(port=0)
                
                with open('token.pickle', 'wb') as token:
                    pickle.dump(creds, token)

            self.service = build('gmail', 'v1', credentials=creds)
            logger.info("Successfully authenticated with Gmail API")
            return True
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            return False

    def get_gmail_labels(self):
        """Get all Gmail labels"""
        try:
            if not self.labels:
                results = self.service.users().labels().list(userId='me').execute()
                self.labels = results.get('labels', [])
            return self.labels
        except Exception as e:
            print(f"Error fetching labels: {e}")
            return []

    def validate_label(self, folder_name):
        """Validate if the label exists and return its ID"""
        labels = self.get_gmail_labels()
        for label in labels:
            if label['name'].lower() == folder_name.lower():
                return label['id']
        return None

    def create_label_if_not_exists(self, folder_name):
        """Create a new label if it doesn't exist"""
        label_id = self.validate_label(folder_name)
        if label_id:
            return label_id

        try:
            label_object = {
                'name': folder_name,
                'labelListVisibility': 'labelShow',
                'messageListVisibility': 'show'
            }
            created_label = self.service.users().labels().create(
                userId='me', body=label_object).execute()
            print(f"\nCreated new label: {folder_name}")
            return created_label['id']
        except Exception as e:
            print(f"Error creating label: {e}")
            return None

    def get_emails(self, folder_name):
        """Get emails from specified Gmail folder"""
        try:
            print(f"\nChecking label: {folder_name}")
            
            # Validate or create label
            label_id = self.validate_label(folder_name)
            if not label_id:
                print(f"\nLabel '{folder_name}' not found. Creating it...")
                label_id = self.create_label_if_not_exists(folder_name)
                if not label_id:
                    return []
                
                print("\nTo automatically organize resumes:")
                print("1. Go to Gmail Settings")
                print("2. Go to 'Filters and Blocked Addresses'")
                print("3. Click 'Create a new filter'")
                print("4. Check 'Has attachment'")
                print("5. Click 'Create filter'")
                print(f"6. Check 'Apply the label' and select '{folder_name}'")
            
            # Get messages with the specified label
            results = self.service.users().messages().list(
                userId='me', labelIds=[label_id]).execute()
            messages = results.get('messages', [])
            
            if not messages:
                print(f"\nNo emails found in label '{folder_name}'")
                print("\nTo add emails to this label:")
                print("1. Open the email with resume")
                print("2. Click the label icon (folder icon)")
                print(f"3. Select '{folder_name}'")
                return []
            
            print(f"Found {len(messages)} emails in label '{folder_name}'")
            return messages
            
        except Exception as e:
            print(f"\nError fetching emails: {e}")
            print("Please check if:")
            print("1. The label name is correct")
            print("2. You have granted the necessary permissions")
            return []

    def extract_email_address(self, message):
        """Extract email address from message headers"""
        try:
            msg = self.service.users().messages().get(
                userId='me', id=message['id'], format='metadata',
                metadataHeaders=['From', 'Subject']).execute()
            
            for header in msg['payload']['headers']:
                if header['name'] == 'From':
                    # Extract email from "Name <email@domain.com>" format
                    email_match = re.search(r'[\w\.-]+@[\w\.-]+', header['value'])
                    if email_match:
                        return email_match.group(0)
            return None
        except Exception:
            return None

    def extract_attachments(self, message_id):
        """Extract attachments from email"""
        try:
            message = self.service.users().messages().get(
                userId='me', id=message_id).execute()
            
            attachments = []
            if 'parts' in message['payload']:
                for part in message['payload']['parts']:
                    if part.get('filename'):
                        if 'data' in part['body']:
                            data = part['body']['data']
                        else:
                            att_id = part['body']['attachmentId']
                            att = self.service.users().messages().attachments().get(
                                userId='me', messageId=message_id, id=att_id).execute()
                            data = att['data']
                        
                        file_data = base64.urlsafe_b64decode(data)
                        attachments.append({
                            'filename': part['filename'],
                            'data': file_data
                        })
            return attachments
        except Exception as e:
            print(f"Error extracting attachments: {e}")
            return []

    def extract_text_from_resume(self, file_data, filename):
        """Extract text from resume file with improved error handling"""
        try:
            logger.info(f"Extracting text from {filename}")
            if filename.lower().endswith('.pdf'):
                return self._extract_from_pdf(file_data)
            elif filename.lower().endswith('.docx'):
                return self._extract_from_docx(file_data)
            else:
                logger.warning(f"Unsupported file format: {filename}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {e}")
            return ""

    def _extract_from_pdf(self, file_data):
        """Extract text from PDF with improved error handling"""
        try:
            pdf_file = BytesIO(file_data)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            extracted_text = text.strip()
            logger.info(f"Successfully extracted {len(extracted_text)} characters from PDF")
            return extracted_text
        except Exception as e:
            logger.error(f"Error extracting from PDF: {e}")
            return ""

    def _extract_from_docx(self, file_data):
        """Extract text from DOCX with improved error handling"""
        try:
            docx_file = BytesIO(file_data)
            doc = docx.Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            extracted_text = text.strip()
            logger.info(f"Successfully extracted {len(extracted_text)} characters from DOCX")
            return extracted_text
        except Exception as e:
            logger.error(f"Error extracting from DOCX: {e}")
            return ""

    def analyze_candidates(self, job_description, resumes):
        """Analyze candidates based on job description with progress tracking"""
        logger.info("Starting candidate analysis...")
        print("\nAnalyzing resumes against job description...")
        
        results = []
        for resume in tqdm(resumes, desc="Processing resumes"):
            try:
                # Log resume content length for debugging
                logger.info(f"Processing resume: {resume['filename']} ({len(resume['text'])} characters)")
                
                # Analyze with LangChain
                analysis = self.analyzer.analyze_resume(resume['text'], job_description)
                
                results.append({
                    'email': resume['email'],
                    'filename': resume['filename'],
                    'similarity_score': analysis['score'],
                    'analysis': analysis['analysis'],
                    'resume_text': resume['text']
                })
            except Exception as e:
                logger.error(f"Error processing resume {resume['filename']}: {e}")
                results.append({
                    'email': resume['email'],
                    'filename': resume['filename'],
                    'similarity_score': 0,
                    'analysis': f"Error during analysis: {str(e)}",
                    'resume_text': resume['text']
                })
        
        # Sort results and add ranking
        df = pd.DataFrame(results)
        df = df.sort_values('similarity_score', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        
        # Save detailed results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f'resume_analysis_{timestamp}.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Analysis results saved to {output_file}")
        
        return df

    def print_results(self, results_df):
        """Print analysis results in a more readable format"""
        print("\n=== Resume Analysis Results ===\n")
        
        for _, row in results_df.iterrows():
            print(f"\nRank #{row['rank']} - Score: {row['similarity_score']}%")
            print(f"Email: {row['email']}")
            print(f"Resume: {row['filename']}")
            print("\nAnalysis:")
            print("-" * 50)
            print(row['analysis'])
            print("=" * 50)

    def process_emails(self, folder_name, job_description):
        """Process emails from Gmail with improved error handling"""
        try:
            if not self.service:
                if not self.authenticate():
                    return pd.DataFrame()

            # Get the label ID
            label_id = None
            results = self.service.users().labels().list(userId='me').execute()
            for label in results.get('labels', []):
                if label['name'].lower() == folder_name.lower():
                    label_id = label['id']
                    break

            if not label_id:
                logger.error(f"Label '{folder_name}' not found")
                return pd.DataFrame()

            # Get messages with the label
            messages = self.service.users().messages().list(
                userId='me',
                labelIds=[label_id]
            ).execute()

            resumes = []
            for message in messages.get('messages', []):
                try:
                    msg = self.service.users().messages().get(
                        userId='me',
                        id=message['id']
                    ).execute()

                    # Get email address
                    headers = msg['payload']['headers']
                    email = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown')

                    # Process attachments
                    if 'parts' in msg['payload']:
                        for part in msg['payload']['parts']:
                            if part.get('filename'):
                                if part['filename'].lower().endswith(('.pdf', '.docx')):
                                    attachment_id = part['body']['attachmentId']
                                    attachment = self.service.users().messages().attachments().get(
                                        userId='me',
                                        messageId=message['id'],
                                        id=attachment_id
                                    ).execute()

                                    file_data = base64.urlsafe_b64decode(attachment['data'])
                                    text = self.extract_text_from_resume(file_data, part['filename'])
                                    
                                    if text:
                                        resumes.append({
                                            'email': email,
                                            'filename': part['filename'],
                                            'text': text
                                        })
                                        logger.info(f"Successfully processed resume: {part['filename']}")

                except Exception as e:
                    logger.error(f"Error processing message {message['id']}: {e}")
                    continue

            if resumes:
                results = self.analyze_candidates(job_description, resumes)
                self.print_results(results)
                return results
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error in process_emails: {e}")
            return pd.DataFrame()

def print_welcome():
    print("\n" + "="*50)
    print("Welcome to HR Recruiter Resume Analyzer")
    print("="*50)
    print("\nBefore starting, please ensure you have:")
    print("1. Set up Google Cloud Project and downloaded credentials.json")
    print("2. Prepared your job description")
    print("\n" + "="*50)

def get_job_description():
    print("\nEnter the Job Description:")
    print("(You can copy-paste the entire JD here)")
    print("Press Enter twice when done")
    print("-"*50)
    
    lines = []
    while True:
        try:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        except EOFError:
            break
    
    job_description = "\n".join(lines)
    print("\nJob Description received. Length:", len(job_description))
    return job_description

def main():
    processor = GmailResumeProcessor()
    
    # Get folder name
    folder_name = input("\nEnter the Gmail label name for resumes:\n(e.g., 'Resumes' or 'Job Applications')\n")
    
    # Get job description
    print("\nEnter the Job Description:")
    print("(You can copy-paste the entire JD here)")
    print("Press Enter twice when done")
    print("-" * 50)
    
    job_description_lines = []
    while True:
        line = input()
        if line == "" and job_description_lines and job_description_lines[-1] == "":
            break
        job_description_lines.append(line)
    
    job_description = "\n".join(job_description_lines)
    print(f"\nJob Description received. Length: {len(job_description)}")
    
    print("\nStarting analysis...")
    processor.process_emails(folder_name, job_description)

if __name__ == "__main__":
    print_welcome()
    main() 