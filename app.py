# app.py
import asyncio
# Ensure an event loop exists for gRPC async clients (fixes Streamlit thread issue)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import os
import json
import re
import time
import uuid
import base64
from typing import TypedDict, Annotated, List, Optional

# LangChain & LangGraph
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Tools & LLMs (Gemini)
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Google API
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText

from dotenv import load_dotenv
load_dotenv()

# --------------------------
# Configuration (put these in .env, no `export`)
# --------------------------
# QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=
# QDRANT_COLLECTION=presentation_memory
# GOOGLE_API_KEY=your_google_api_key_here

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "presentation_memory")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)

# --------------------------
# Validate keys early
# --------------------------
if GOOGLE_API_KEY is None or GOOGLE_API_KEY.strip() == "":
    # Not fatal: we surface helpful error when attempting to use Gemini/embeddings
    st_warning = None  # placeholder so this file imports in non-streamlit contexts
    # We'll raise later when embeddings/llm are created to give actionable message.

# --------------------------
# Initialize Qdrant client
# --------------------------
try:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to initialize QdrantClient (url={QDRANT_URL}). Error: {e}") from e

# --------------------------
# Embeddings (Google Generative AI)
# --------------------------
try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )
except Exception as e:
    # Provide a clear runtime error so the user knows to set GOOGLE_API_KEY or install packages
    raise RuntimeError(
        "Failed to initialize GoogleGenerativeAIEmbeddings. "
        "Make sure `langchain_google_genai` is installed and GOOGLE_API_KEY is set in your .env. "
        f"Original error: {e}"
    ) from e

# --------------------------
# Ensure Qdrant collection exists (non-deprecated approach)
# --------------------------
def ensure_collection(collection_name: str):
    # Prefer collection_exists if available; otherwise fallback to get_collections
    try:
        exists = qdrant_client.collection_exists(collection_name)
    except Exception:
        exists = False
        try:
            cols = qdrant_client.get_collections()
            exists = any(c.name == collection_name for c in cols.collections)
        except Exception:
            exists = False

    if not exists:
        # generate a sample vector to find embedding size:
        sample_vec = embeddings.embed_query("initialize vector sizing")
        size = len(sample_vec)
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE)
        )

ensure_collection(COLLECTION_NAME)

# --------------------------
# Qdrant helpers: upsert + search
# --------------------------
def qdrant_upsert_session(topic: str,
                          structured_content: Optional[str] = None,
                          refined_content: Optional[str] = None,
                          recipient_email: Optional[str] = None,
                          presentation_id: Optional[str] = None,
                          session_id: Optional[str] = None) -> str:
    """
    Upsert a session into Qdrant; returns the session id used.
    """
    text_to_embed = refined_content or structured_content or topic
    vector = embeddings.embed_query(text_to_embed)
    sid = session_id or str(uuid.uuid4())

    payload = {
        "topic": topic,
        "structured_content": structured_content,
        "refined_content": refined_content,
        "recipient_email": recipient_email,
        "presentation_id": presentation_id,
        "timestamp": int(time.time()),
    }

    point = PointStruct(id=sid, vector=vector, payload=payload)
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])
    return sid

def qdrant_search_topic(topic_query: str, top_k: int = 3):
    qv = embeddings.embed_query(topic_query)
    hits = qdrant_client.search(collection_name=COLLECTION_NAME, query_vector=qv, limit=top_k)
    return hits

# --------------------------
# Google auth helper
# --------------------------
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

# --------------------------
# Agent state & nodes
# --------------------------
class AgentState(TypedDict):
    topic: str
    structured_content: str
    refined_content: str
    presentation_id: str
    recipient_email: Optional[str]
    qdrant_session_id: Optional[str]
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

def research_node(state: AgentState):
    topic = state.get("topic")
    if not topic:
        raise ValueError("Missing 'topic' in state")

    st.info("🔍 Researching topic...")
    search = DuckDuckGoSearchRun()
    search_results = search.run(f"in-depth overview of {topic} for a professional presentation")

    st.info("📝 Structuring content with Gemini...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=GOOGLE_API_KEY)

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
    structured = response.content

    sid = qdrant_upsert_session(topic=topic, structured_content=structured)
    return {"structured_content": structured, "topic": topic, "qdrant_session_id": sid}

def refinement_node(state: AgentState):
    st.info("✨ Refining slide content...")
    raw_json = state["structured_content"]
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, google_api_key=GOOGLE_API_KEY)

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
    refined_text = refined.content

    sid = state.get("qdrant_session_id")
    sid = qdrant_upsert_session(topic=state["topic"],
                               structured_content=raw_json,
                               refined_content=refined_text,
                               session_id=sid)
    return {"refined_content": refined_text, "topic": state["topic"], "qdrant_session_id": sid}

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
                if placeholder.get('type') == 'TITLE':
                    title_id = shape['objectId']
                if placeholder.get('type') == 'BODY':
                    body_id = shape['objectId']

            text_requests = []
            if title_id:
                text_requests.append({'insertText': {'objectId': title_id, 'text': slide_title}})
            if body_id:
                text_requests.append({'insertText': {'objectId': body_id, 'text': slide_content}})

            if text_requests:
                slides_service.presentations().batchUpdate(
                    presentationId=presentation_id,
                    body={'requests': text_requests}
                ).execute()

        sid = state.get("qdrant_session_id")
        qdrant_upsert_session(topic=state["topic"],
                              structured_content=state.get("structured_content"),
                              refined_content=state.get("refined_content"),
                              recipient_email=state.get("recipient_email"),
                              presentation_id=presentation_id,
                              session_id=sid)

        return {"presentation_id": presentation_id, "topic": state["topic"], "qdrant_session_id": sid}
    except Exception as e:
        st.error(f"Error creating presentation: {e}")
        return {"presentation_id": f"Error: {e}", "topic": state.get("topic", ""), "qdrant_session_id": state.get("qdrant_session_id")}

def send_email_node(state: AgentState):
    st.info("📧 Sending email...")
    recipient_email = state.get("recipient_email")
    if not recipient_email:
        hits = qdrant_search_topic(state["topic"], top_k=1)
        if hits:
            payload = hits[0].payload or {}
            recipient_email = payload.get("recipient_email")

    if not recipient_email:
        return {"error": "Missing recipient_email; could not find in state or Qdrant."}

    presentation_id = state.get("presentation_id")
    if not presentation_id:
        return {"error": "Missing presentation_id in state."}

    presentation_url = f"https://docs.google.com/presentation/d/{presentation_id}/edit"
    subject = "fhir" if "fhir" in state["topic"].lower() else f"Presentation: {state['topic']}"
    body = f"Hello,\n\nThis is to notify you that I have sent the presentation on '{state['topic']}'.\n\nYou can view it here: {presentation_url}\n\nBest regards."

    creds = authenticate_google_services()
    gmail_service = build("gmail", "v1", credentials=creds)

    message = MIMEText(body)
    message["to"] = recipient_email
    message["subject"] = subject
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

    gmail_service.users().messages().send(userId="me", body={"raw": raw_message}).execute()

    qdrant_upsert_session(topic=state["topic"],
                          structured_content=state.get("structured_content"),
                          refined_content=state.get("refined_content"),
                          recipient_email=recipient_email,
                          presentation_id=presentation_id,
                          session_id=state.get("qdrant_session_id"))

    return {"sent_to": recipient_email, "presentation_id": presentation_id}

# --------------------------
# Build workflow
# --------------------------
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

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="AI Presentation Agent", page_icon="📊")
st.title("📊 AI Research & Presentation Agent (Qdrant-backed)")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": st.session_state.thread_id}}

topic = st.text_input("Enter a research topic:")
recipient = st.text_input("Recipient's Email Address (optional — will be saved to Qdrant)")

if st.button("Generate & Send Presentation"):
    if not topic:
        st.warning("Please provide a topic.")
    else:
        initial_state = {"topic": topic, "messages": [HumanMessage(content=topic)], "recipient_email": recipient or None}
        with st.spinner("Agent is working..."):
            for _ in app.stream(initial_state, config=config):
                pass
        st.success("Workflow complete. If an email was requested it was sent (or recorded).")
        st.balloons()

st.markdown("---")
