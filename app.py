import streamlit as st
import os
import json
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
import re
import time
import uuid

# Google API Imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText
import base64

from dotenv import load_dotenv
load_dotenv()

# --- Configuration and Authentication ---
SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/presentations",
    "https://www.googleapis.com/auth/drive.file"
]

def authenticate_google_services():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return creds

# --- Agent State Definition ---
class AgentState(TypedDict):
    topic: str
    structured_content: str
    refined_content: str
    presentation_id: str
    recipient_email: Optional[str]
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# --- Tool Nodes ---
def research_node(state: AgentState):
    topic = state.get("topic")
    if not topic:
        raise ValueError("Missing 'topic' in state")

    st.info("🔍 Researching topic...")
    search = DuckDuckGoSearchRun()
    search_results = search.run(f"in-depth overview of {topic} for a professional presentation")

    st.info("📝 Structuring content with Gemini...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=os.getenv("GOOGLE_API_KEY"))

    prompt = f"""
    Based on the following research, create a JSON structure for a professional, 10-slide presentation on '{topic}'.
    The JSON must have a "presentation_title" and a "slides" list.
    Each slide object in the list must have:
    1. A "title" (string).
    2. "content" (a list of concise bullet points).
    3. "speaker_notes" (a string suggesting a highly relevant visual).

    IMPORTANT: Respond with ONLY the raw JSON, no markdown or explanation.

    Research Material:
    {search_results}
    """
    response = llm.invoke(prompt)
    return {"structured_content": response.content, "topic": topic}

def refinement_node(state: AgentState):
    st.info("✨ Refining slide content...")
    raw_json = state["structured_content"]

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, google_api_key=os.getenv("GOOGLE_API_KEY"))
    refine_prompt = f"""
You are a professional presentation designer.
Refine the following JSON into polished slide content:
- Titles ≤ 8 words
- Bullets ≤ 8 words
- Speaker notes suggest visuals

Keep JSON schema the same. Only rewrite text.

Raw JSON:
{raw_json}
"""
    refined = llm.invoke(refine_prompt)
    return {"refined_content": refined.content, "topic": state["topic"]}

def create_presentation_node(state: AgentState):
    st.info("📊 Creating Google Slides presentation...")
    creds = authenticate_google_services()
    slides_service = build("slides", "v1", credentials=creds)
    drive_service = build("drive", "v3", credentials=creds)

    try:
        cleaned_json_string = re.sub(r'```json\s*|\s*```', '', state['refined_content'].strip())
        data = json.loads(cleaned_json_string)

        presentation_title = data.get("presentation_title", "AI-Generated Presentation")
        presentation_body = {'name': presentation_title, 'mimeType': 'application/vnd.google-apps.presentation'}
        presentation_file = drive_service.files().create(body=presentation_body).execute()
        presentation_id = presentation_file.get('id')

        # Make shareable
        drive_service.permissions().create(
            fileId=presentation_id,
            body={"role": "reader", "type": "anyone"}
        ).execute()

        # Remove default slide
        presentation = slides_service.presentations().get(presentationId=presentation_id).execute()
        default_slide_id = presentation['slides'][0]['objectId']
        requests = [{'deleteObject': {'objectId': default_slide_id}}]
        slides_service.presentations().batchUpdate(presentationId=presentation_id, body={'requests': requests}).execute()

        # Create slides
        for i, slide_info in enumerate(data.get("slides", [])):
            slide_title = slide_info.get("title", "")
            slide_content = "\n".join([f"• {item}" for item in slide_info.get("content", [])])
            speaker_notes = slide_info.get("speaker_notes", "No visual suggested.")

            create_slide_request = {
                'createSlide': {
                    'objectId': f'slide_{i}',
                    'slideLayoutReference': {'predefinedLayout': 'TITLE_AND_BODY'}
                }
            }
            response = slides_service.presentations().batchUpdate(
                presentationId=presentation_id,
                body={'requests': [create_slide_request]}
            ).execute()
            new_slide_id = response['replies'][0]['createSlide']['objectId']

            time.sleep(1)

            new_slide_details = slides_service.presentations().pages().get(
                presentationId=presentation_id,
                pageObjectId=new_slide_id
            ).execute()
            title_id, body_id = None, None
            for shape in new_slide_details.get('pageElements', []):
                placeholder = shape.get('shape', {}).get('placeholder', {})
                if placeholder.get('type') == 'TITLE': title_id = shape['objectId']
                if placeholder.get('type') == 'BODY': body_id = shape['objectId']

            text_requests = []
            if title_id: text_requests.append({'insertText': {'objectId': title_id, 'text': slide_title}})
            if body_id: text_requests.append({'insertText': {'objectId': body_id, 'text': slide_content}})

            if text_requests:
                slides_service.presentations().batchUpdate(
                    presentationId=presentation_id,
                    body={'requests': text_requests}
                ).execute()

        return {"presentation_id": presentation_id, "topic": state["topic"]}
    except Exception as e:
        return {"presentation_id": f"Error: {e}", "topic": state.get("topic", "")}

def send_email_node(state: AgentState):
    st.info("📧 Sending email...")
    if 'presentation_id' not in state or 'recipient_email' not in state:
        return {"error": "Missing presentation_id or recipient_email in state."}

    presentation_id = state['presentation_id']
    recipient_email = state['recipient_email']
    presentation_url = f"https://docs.google.com/presentation/d/{presentation_id}/edit"

    subject = f"Presentation on: {state['topic']}"
    body = f"Hello,\n\nHere is the AI-generated presentation on '{state['topic']}'.\n\nView it here: {presentation_url}"

    creds = authenticate_google_services()
    gmail_service = build("gmail", "v1", credentials=creds)

    message = MIMEText(body)
    message["to"] = recipient_email
    message["subject"] = subject
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

    gmail_service.users().messages().send(userId="me", body={"raw": raw_message}).execute()
    return {}

# --- Graph Definition ---
workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("refinement", refinement_node)
workflow.add_node("create_presentation", create_presentation_node)
workflow.add_node("send_email", send_email_node)

workflow.set_entry_point("research")
workflow.add_edge("research", "refinement")
workflow.add_edge("refinement", "create_presentation")
workflow.add_edge("create_presentation", "send_email")
workflow.add_edge("send_email", END)

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# --- Streamlit UI ---
st.set_page_config(page_title="AI Presentation Agent", page_icon="📊")
st.title("📊 AI Research & Presentation Agent")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": st.session_state.thread_id}}

topic = st.text_input("Enter a research topic:")
recipient = st.text_input("Recipient's Email Address")

if st.button("Generate & Send Presentation"):
    if topic and recipient:
        initial_state = {"topic": topic, "messages": [HumanMessage(content=topic)], "recipient_email": recipient}
        with st.spinner("Agent is working..."):
            for _ in app.stream(initial_state, config=config):
                pass
        st.success(f"✅ Email sent successfully to {recipient}!")
        st.balloons()
