"""
ATLAS: Academic Task & Learning Agent System
Complete implementation with all features in a single file.

Features:
- PDF Analysis with RAG
- Web Research with Tavily
- Notion Integration (CRUD operations)
- Task Planning and Scheduling
- Academic Advisor with personalization
- Voice Input/Output
- Multi-agent orchestration with sequential execution
"""

import os
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import streamlit as st
import torch
import tempfile
import base64
import time
import requests
import json
import asyncio
import operator
import logging
from typing import TypedDict, List, Dict, Any, Union, Annotated, Optional
from dotenv import load_dotenv
from gtts import gTTS
from pydantic import BaseModel, Field

# Langchain & LangGraph Core
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import StateGraph, END

# Langchain Community & Integrations
from langchain.agents import create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.tools.tavily_search import TavilySearchResults

# LLM Providers
from langchain_cerebras import ChatCerebras
from langchain_openai import ChatOpenAI

# ==============================================================================
# CONFIGURATION AND SETUP
# ==============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# API Keys
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_NOTES_DB_ID = os.getenv("DATABASE_ID")
NOTION_CALENDAR_DB_ID = os.getenv("DATABASE_ID1")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVER_K = 5
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SIZE = 512
AUDIO_SILENCE_SECONDS = 2
AUDIO_TIMEOUT_SECONDS = 30
VAD_THRESHOLD = 0.5
MAX_SEARCH_RESULTS = 5

# Notion API setup
NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}
NOTION_API_BASE = "https://api.notion.com/v1"

# Initialize LLM
try:
    llm = ChatCerebras(model="llama-4-scout-17b-16e-instruct", api_key=CEREBRAS_API_KEY)
    logger.info("Successfully initialized Cerebras LLM.")
except Exception as e:
    logger.error(f"Cerebras LLM initialization failed: {e}")
    st.warning("Could not initialize Cerebras LLM. Falling back to OpenAI.")
    try:
        llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
        logger.info("Successfully initialized OpenAI LLM as fallback.")
    except Exception as openai_e:
        logger.error(f"OpenAI LLM initialization failed: {openai_e}")
        st.error(f"FATAL: Could not initialize any LLM. Error: {openai_e}")
        st.stop()

# ==============================================================================
# VOICE HELPERS
# ==============================================================================

audio_queue = queue.Queue()

@st.cache_resource
def load_vad_model():
    model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
    return model

def audio_callback(indata, frames, time_info, status):
    if status:
        logger.warning(f"Audio callback status: {status}")
    audio_queue.put(indata.copy())

def record_audio(filename="temp_recording.wav"):
    vad_model = load_vad_model()
    st.info("🎤 Listening... Please speak now.")
    audio_data, recorded, silent_chunks = [], False, 0
    max_silent_chunks = int(AUDIO_SILENCE_SECONDS * AUDIO_SAMPLE_RATE / AUDIO_CHUNK_SIZE)
    max_chunks = int(AUDIO_TIMEOUT_SECONDS * AUDIO_SAMPLE_RATE / AUDIO_CHUNK_SIZE)

    with sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=AUDIO_CHUNK_SIZE):
        for _ in range(max_chunks):
            chunk = audio_queue.get()
            audio_data.append(chunk)
            speech_prob = vad_model(torch.from_numpy(chunk.flatten()).float(), AUDIO_SAMPLE_RATE).item()
            if speech_prob > VAD_THRESHOLD:
                silent_chunks = 0
                recorded = True
            elif recorded:
                silent_chunks += 1
            if recorded and silent_chunks > max_silent_chunks:
                st.success("✅ Silence detected, processing...")
                break
        else:
            if recorded:
                st.success("✅ Recording timed out, processing...")
    
    if not recorded:
        return False
    sf.write(filename, np.concatenate(audio_data, axis=0), AUDIO_SAMPLE_RATE)
    return True

def transcribe_audio(file_path):
    if not DEEPGRAM_API_KEY:
        st.error("Deepgram API key not set.")
        return ""
    url = "https://api.deepgram.com/v1/listen"
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    try:
        with open(file_path, "rb") as f:
            audio_data = f.read()
        response = requests.post(url, headers=headers, data=audio_data, timeout=30)
        response.raise_for_status()
        return response.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
    except Exception as e:
        st.error(f"❌ Transcription error: {e}")
        return ""

def speak_text(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            return tmp.name
    except Exception as e:
        st.error(f"TTS error: {e}")
        return None

def play_audio(file_path):
    if file_path and os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            audio_bytes = f.read()
        b64_audio = base64.b64encode(audio_bytes).decode()
        st.components.v1.html(f'<audio autoplay><source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3"></audio>', height=50)
        time.sleep(1)
        os.remove(file_path)

# ==============================================================================
# NOTION AGENT TOOLS
# ==============================================================================

@tool
def add_note_to_database(name: str, notes: str, link_url: Optional[str] = None) -> str:
    """Adds a new note to Notion database with name, content, and optional URL."""
    if not NOTION_NOTES_DB_ID:
        return "Error: Notion Notes Database ID is not configured."
    url = f"{NOTION_API_BASE}/pages"
    properties = {
        "Name": {"title": [{"text": {"content": name}}]},
        "NOTES": {"rich_text": [{"type": "text", "text": {"content": notes}}]}
    }
    if link_url:
        properties["url"] = {"url": link_url}
    payload = {"parent": {"database_id": NOTION_NOTES_DB_ID}, "properties": properties}
    r = requests.post(url, headers=NOTION_HEADERS, json=payload)
    return f"✅ Note '{name}' added successfully." if r.ok else f"❌ Failed: {r.status_code} - {r.text}"

@tool
def schedule_calendar_event(
    title: str,
    date: str,
    description: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    timezone: Optional[str] = "UTC"
) -> str:
    """Schedules an event in Notion calendar. Date: YYYY-MM-DD, Times: HH:MM format."""
    if not NOTION_CALENDAR_DB_ID:
        return "Error: Notion Calendar Database ID is not configured."
    url = f"{NOTION_API_BASE}/pages"
    date_property = {"start": date}
    if start_time:
        date_property["start"] = f"{date}T{start_time}:00"
        if end_time:
            date_property["end"] = f"{date}T{end_time}:00"
    properties = {
        "Title": {"title": [{"text": {"content": title}}]},
        "Date": {"date": date_property}
    }
    if description:
        properties["description"] = {"rich_text": [{"text": {"content": description}}]}
    payload = {"parent": {"database_id": NOTION_CALENDAR_DB_ID}, "properties": properties}
    r = requests.post(url, headers=NOTION_HEADERS, json=payload)
    return f"✅ Event '{title}' scheduled for {date}." if r.ok else f"❌ Failed: {r.status_code}"

@tool
def query_notion_database(query: str) -> str:
    """Searches Notion notes database for pages matching the query text."""
    if not NOTION_NOTES_DB_ID:
        return "Error: Notion Notes DB ID not configured."
    url = f"{NOTION_API_BASE}/databases/{NOTION_NOTES_DB_ID}/query"
    payload = {"filter": {"property": "Name", "title": {"contains": query}}}
    r = requests.post(url, headers=NOTION_HEADERS, json=payload)
    if not r.ok:
        return f"❌ Failed: {r.status_code}"
    results = r.json().get("results", [])
    if not results:
        return f"No pages found for: {query}"
    return "Found pages:\n" + "\n".join([f"- {res['properties']['Name']['title'][0]['text']['content']} (ID: {res['id']})" for res in results])

@tool
def update_notion_page(page_id: str, content: str) -> str:
    """Appends content to an existing Notion page."""
    url = f"{NOTION_API_BASE}/blocks/{page_id}/children"
    payload = {"children": [{"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": content}}]}}]}
    r = requests.patch(url, headers=NOTION_HEADERS, json=payload)
    return "✅ Page updated successfully." if r.ok else f"❌ Failed: {r.status_code}"

@tool
def get_notion_page_content(page_id: str) -> str:
    """Reads all text content from a Notion page."""
    url = f"{NOTION_API_BASE}/blocks/{page_id}/children"
    try:
        response = requests.get(url, headers=NOTION_HEADERS)
        response.raise_for_status()
        data = response.json()
        content = []
        supported_blocks = ["paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item", "to_do", "toggle", "callout", "quote"]
        for block in data.get("results", []):
            block_type = block.get("type")
            if block_type in supported_blocks:
                text_parts = [text_item["plain_text"] for text_item in block[block_type].get("rich_text", [])]
                content.append("".join(text_parts))
        full_content = "\n".join(content)
        return f"Content:\n\n{full_content}" if full_content else f"No content found."
    except Exception as e:
        return f"❌ Failed: {e}"

@tool
def delete_notion_page(page_id: str) -> str:
    """Archives (deletes) a Notion page."""
    url = f"{NOTION_API_BASE}/pages/{page_id}"
    payload = {"archived": True}
    r = requests.patch(url, headers=NOTION_HEADERS, json=payload)
    return f"✅ Page archived." if r.ok else f"❌ Failed: {r.status_code}"

notion_tools = [add_note_to_database, schedule_calendar_event, query_notion_database, update_notion_page, get_notion_page_content, delete_notion_page]

# ==============================================================================
# WEB RESEARCHER TOOLS
# ==============================================================================

@tool
def search_the_web(query: str, max_results: int = MAX_SEARCH_RESULTS) -> str:
    """Performs web search and returns summarized results."""
    if not TAVILY_API_KEY:
        return "Error: Tavily API key not configured."
    try:
        search = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=max_results)
        results = search.invoke(query)
        if not results:
            return f"No results found for: '{query}'"
        output = f"Search results for '{query}':\n\n"
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('url', 'No URL')
            content = result.get('content', 'No description')
            output += f"{i}. **{title}**\n   URL: {url}\n   {content}\n\n"
        return output
    except Exception as e:
        return f"❌ Error: {e}"

@tool
def scrape_website(url: str) -> str:
    """Scrapes text content from a website URL."""
    try:
        loader = WebBaseLoader(url)
        loader.requests_kwargs = {'timeout': 30}
        docs = loader.load()
        if not docs:
            return f"No content from {url}"
        content = "\n\n".join([doc.page_content for doc in docs])
        if len(content) > 5000:
            content = content[:5000] + "\n\n[Truncated...]"
        return f"Content from {url}:\n\n{content}"
    except Exception as e:
        return f"❌ Error: {e}"

web_tools = [search_the_web, scrape_website]

# ==============================================================================
# PLANNER TOOLS
# ==============================================================================

@tool
def create_todo_list(steps: List[str]) -> str:
    """Creates a markdown checklist from task steps."""
    if not steps:
        return "Error: No steps provided."
    todo_list = "# Task Checklist\n\n"
    for i, step in enumerate(steps, 1):
        todo_list += f"- [ ] {step}\n"
    return todo_list

@tool
def break_down_task(task: str) -> str:
    """Breaks down a complex task into manageable steps using LLM."""
    if not task:
        return "Error: No task provided."
    prompt = f"Break down this task into clear, actionable steps:\n\nTask: {task}\n\nProvide a numbered list."
    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"❌ Error: {e}"

@tool
def create_study_schedule(subjects: List[str], hours_per_day: int = 4, days: int = 7) -> str:
    """Creates a study schedule for multiple subjects."""
    if not subjects:
        return "Error: No subjects provided."
    if hours_per_day <= 0 or days <= 0:
        return "Error: Invalid parameters."
    hours_per_subject = hours_per_day / len(subjects)
    schedule = f"# Study Schedule ({days} days)\n\n**Total**: {hours_per_day} hours/day\n**Subjects**: {', '.join(subjects)}\n\n"
    for day in range(1, days + 1):
        schedule += f"## Day {day}\n\n"
        for subject in subjects:
            schedule += f"- **{subject}**: {hours_per_subject:.1f} hours\n"
        schedule += "\n"
    return schedule

@tool
def prioritize_tasks(tasks: List[str], criteria: str = "urgency and importance") -> str:
    """Prioritizes tasks based on criteria."""
    if not tasks:
        return "Error: No tasks provided."
    high_keywords = ['urgent', 'critical', 'important', 'deadline', 'exam', 'test']
    medium_keywords = ['soon', 'assignment', 'homework', 'project']
    high, medium, low = [], [], []
    for task in tasks:
        task_lower = task.lower()
        if any(kw in task_lower for kw in high_keywords):
            high.append(task)
        elif any(kw in task_lower for kw in medium_keywords):
            medium.append(task)
        else:
            low.append(task)
    output = f"# Task Prioritization\n\n"
    if high:
        output += "## 🔴 High Priority\n" + "\n".join([f"- {t}" for t in high]) + "\n\n"
    if medium:
        output += "## 🟡 Medium Priority\n" + "\n".join([f"- {t}" for t in medium]) + "\n\n"
    if low:
        output += "## 🟢 Low Priority\n" + "\n".join([f"- {t}" for t in low])
    return output

planner_tools = [create_todo_list, break_down_task, create_study_schedule, prioritize_tasks]

# ==============================================================================
# NOTION AGENT GRAPH
# ==============================================================================

class NotionAgentState(TypedDict):
    question: str
    agent_outcome: Union[AgentAction, AgentFinish, List[AgentAction]]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    chat_history: List[BaseMessage]

notion_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Notion assistant. Use these tools:
1. query_notion_database - Find pages by title
2. get_notion_page_content - Read page content by ID
3. add_note_to_database - Create new notes
4. update_notion_page - Add content to existing pages
5. schedule_calendar_event - Schedule events
6. delete_notion_page - Archive pages

Strategy: Search first, then read content if needed."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

notion_agent_runnable = create_tool_calling_agent(llm, notion_tools, notion_prompt)

def run_notion_agent_node(state: NotionAgentState):
    return {"agent_outcome": notion_agent_runnable.invoke(state)}

def execute_notion_tools_node(state: NotionAgentState):
    actions = state.get("agent_outcome") if isinstance(state.get("agent_outcome"), list) else [state.get("agent_outcome")]
    tool_map = {t.name: t for t in notion_tools}
    steps = []
    for action in actions:
        if not isinstance(action, AgentAction):
            continue
        tool = tool_map.get(action.tool)
        if tool:
            try:
                steps.append((action, str(tool.func(**action.tool_input))))
            except Exception as e:
                steps.append((action, f"Error: {e}"))
        else:
            steps.append((action, f"Unknown tool: {action.tool}"))
    return {"intermediate_steps": steps}

def should_continue_notion(state: NotionAgentState):
    return END if isinstance(state["agent_outcome"], AgentFinish) else "tools"

notion_graph = StateGraph(NotionAgentState)
notion_graph.add_node("agent", run_notion_agent_node)
notion_graph.add_node("tools", execute_notion_tools_node)
notion_graph.set_entry_point("agent")
notion_graph.add_conditional_edges("agent", should_continue_notion, {END: END, "tools": "tools"})
notion_graph.add_edge("tools", "agent")
notion_app = notion_graph.compile()

# ==============================================================================
# WEB RESEARCHER GRAPH
# ==============================================================================

class WebResearcherState(TypedDict):
    question: str
    agent_outcome: Union[AgentAction, AgentFinish, List[AgentAction]]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    chat_history: List[BaseMessage]

web_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a web researcher with tools:
1. search_the_web - Search the internet
2. scrape_website - Extract content from URLs

Strategy: Search first, then scrape promising URLs for details."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

web_agent_runnable = create_tool_calling_agent(llm, web_tools, web_prompt)

def run_web_agent_node(state: WebResearcherState):
    return {"agent_outcome": web_agent_runnable.invoke(state)}

def execute_web_tools_node(state: WebResearcherState):
    actions = state.get("agent_outcome") if isinstance(state.get("agent_outcome"), list) else [state.get("agent_outcome")]
    tool_map = {t.name: t for t in web_tools}
    steps = []
    for action in actions:
        if not isinstance(action, AgentAction):
            continue
        tool = tool_map.get(action.tool)
        if tool:
            try:
                steps.append((action, str(tool.func(**action.tool_input))))
            except Exception as e:
                steps.append((action, f"Error: {e}"))
        else:
            steps.append((action, f"Unknown tool"))
    return {"intermediate_steps": steps}

def should_continue_web(state: WebResearcherState):
    return END if isinstance(state["agent_outcome"], AgentFinish) else "tools"

web_graph = StateGraph(WebResearcherState)
web_graph.add_node("agent", run_web_agent_node)
web_graph.add_node("tools", execute_web_tools_node)
web_graph.set_entry_point("agent")
web_graph.add_conditional_edges("agent", should_continue_web, {END: END, "tools": "tools"})
web_graph.add_edge("tools", "agent")
web_app = web_graph.compile()

# ==============================================================================
# PLANNER GRAPH
# ==============================================================================

class PlannerState(TypedDict):
    question: str
    agent_outcome: Union[AgentAction, AgentFinish, List[AgentAction]]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    chat_history: List[BaseMessage]

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a task planning assistant with tools:
1. break_down_task - Decompose complex tasks
2. create_todo_list - Generate checklists
3. create_study_schedule - Plan study time
4. prioritize_tasks - Organize by importance

Help users plan effectively."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

planner_agent_runnable = create_tool_calling_agent(llm, planner_tools, planner_prompt)

def run_planner_node(state: PlannerState):
    return {"agent_outcome": planner_agent_runnable.invoke(state)}

def execute_planner_tools_node(state: PlannerState):
    actions = state.get("agent_outcome") if isinstance(state.get("agent_outcome"), list) else [state.get("agent_outcome")]
    tool_map = {t.name: t for t in planner_tools}
    steps = []
    for action in actions:
        if not isinstance(action, AgentAction):
            continue
        tool = tool_map.get(action.tool)
        if tool:
            try:
                steps.append((action, str(tool.func(**action.tool_input))))
            except Exception as e:
                steps.append((action, f"Error: {e}"))
        else:
            steps.append((action, "Unknown tool"))
    return {"intermediate_steps": steps}

def should_continue_planner(state: PlannerState):
    return END if isinstance(state["agent_outcome"], AgentFinish) else "tools"

planner_graph = StateGraph(PlannerState)
planner_graph.add_node("agent", run_planner_node)
planner_graph.add_node("tools", execute_planner_tools_node)
planner_graph.set_entry_point("agent")
planner_graph.add_conditional_edges("agent", should_continue_planner, {END: END, "tools": "tools"})
planner_graph.add_edge("tools", "agent")
planner_app = planner_graph.compile()

# ==============================================================================
# PDF RESEARCHER (RAG)
# ==============================================================================

@st.cache_data(show_spinner=False)
def get_or_create_vector_store(_pdf_path: str) -> FAISS:
    logger.info(f"Creating vector store for: {_pdf_path}")
    loader = PyMuPDFLoader(_pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_documents(split_docs, embeddings)

async def researcher_rag_node(state: Dict) -> Dict:
    logger.info("Agent: Executing PDF Researcher (RAG)...")
    pdf_path = state.get('pdf_path')
    if not pdf_path or not os.path.exists(pdf_path):
        return {"researcher_output": "Error: PDF not found. Please upload a document."}
    try:
        with st.spinner("📄 Analyzing document..."):
            vector_store = get_or_create_vector_store(pdf_path)
            retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
            doc_qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer based ONLY on this context:\n<context>\n{context}\n</context>"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}")
            ])
            document_chain = create_stuff_documents_chain(llm, doc_qa_prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            result = await retrieval_chain.ainvoke({"input": state['question'], "chat_history": state.get('chat_history', [])})
        return {"researcher_output": result["answer"]}
    except Exception as e:
        logger.error(f"RAG error: {e}", exc_info=True)
        return {"researcher_output": f"❌ Error: {e}"}

# ==============================================================================
# ACADEMIC ADVISOR
# ==============================================================================

class AdvisorState(TypedDict):
    messages: List[BaseMessage]
    profile: Dict[str, Any]
    results: Dict[str, Any]
    chat_history: List[BaseMessage]

class AdvisorAgent:
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        g = StateGraph(AdvisorState)
        g.add_node("analyze", self._analyze_situation)
        g.add_node("generate", self._generate_guidance)
        g.set_entry_point("analyze")
        g.add_edge("analyze", "generate")
        g.add_edge("generate", END)
        return g.compile()
    
    async def _analyze_situation(self, state: AdvisorState) -> Dict:
        profile = state.get("profile", {})
        profile_text = self._format_profile(profile)
        history = "\n".join([f"{m.type}: {m.content}" for m in state.get("chat_history", [])])
        request = state["messages"][-1].content if state.get("messages") else ""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze student's request based on profile and history. Identify key challenges."),
            ("user", f"PROFILE: {profile_text}\nHISTORY: {history}\nREQUEST: {request}\n\nAnalyze:")
        ])
        r = await (prompt | self.llm).ainvoke({})
        return {"results": {"analysis": r.content}}
    
    async def _generate_guidance(self, state: AdvisorState) -> Dict:
        analysis = state.get("results", {}).get("analysis", "")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Provide structured, actionable academic advice."),
            ("user", f"ANALYSIS:\n{analysis}\n\nYour guidance:")
        ])
        r = await (prompt | self.llm).ainvoke({})
        return {"results": {**state["results"], "guidance": r.content}}
    
    def _format_profile(self, profile: Dict) -> str:
        if not profile or not any(profile.values()):
            return "No profile."
        parts = []
        if profile.get("academic_level"):
            parts.append(f"Level: {profile['academic_level']}")
        if profile.get("major"):
            parts.append(f"Major: {profile['major']}")
        if profile.get("interests"):
            parts.append(f"Interests: {profile['interests']}")
        if profile.get("description"):
            parts.append(f"Info: {profile['description']}")
        return "\n".join(parts) if parts else "No profile."
    
    async def __call__(self, state: AdvisorState) -> Dict:
        return await self.workflow.ainvoke(state)

async def advisor_analyze_node(state: Dict, user_profile: Dict = None) -> Dict:
    logger.info("Agent: Executing Academic Advisor...")
    agent = AdvisorAgent(llm=llm)
    advisor_state = AdvisorState(
        messages=[HumanMessage(content=state.get('question', ''))],
        profile=user_profile or {},
        results={},
        chat_history=state.get("chat_history", [])
    )
    result = await agent(advisor_state)
    guidance = result.get("results", {}).get("guidance", "No guidance generated.")
    return {"advisor_output": guidance}

# ==============================================================================
# MAIN ORCHESTRATOR GRAPH
# ==============================================================================

class AgentStep(BaseModel):
    agent: str = Field(description="Agent name")
    query: str = Field(description="Task for agent")

class CoordinatorPlan(BaseModel):
    plan: List[AgentStep] = Field(description="Sequential steps")
    reasoning: str = Field(description="Routing decision explanation")

class AgentState(TypedDict):
    question: str
    pdf_path: Optional[str]
    user_profile: Dict[str, Any]
    coordinator_analysis: dict
    current_step: int
    execution_results: Dict[str, str]
    planner_output: str
    notewriter_output: str
    advisor_output: str
    researcher_output: str
    web_researcher_output: str
    answer: str
    audio_output_path: Optional[str]
    chat_history: List[BaseMessage]

async def coordinator_node(state: AgentState) -> Dict:
    logger.info("Coordinator: Creating execution plan...")
    context_parts = []
    if state.get("pdf_path"):
        context_parts.append("PDF document loaded.")
    if state.get("user_profile") and any(state.get("user_profile", {}).values()):
        profile = state["user_profile"]
        context_parts.append(f"User: {profile.get('academic_level', 'N/A')} student")
    context = " ".join(context_parts) if context_parts else "No context."
    
    prompt_template = """You are a task router. Analyze the query and create an execution plan.

CONTEXT: {context}

AGENTS:
- RESEARCHER: Query PDF (only if loaded)
- WEB_RESEARCHER: Search web for current info
- NOTIONWRITER: Manage Notion notes/events
- PLANNER: Task planning, schedules
- ADVISOR: Academic advice

QUERY: '{query}'

Respond with JSON:
{{
  "plan": [{{"agent": "AGENT_NAME", "query": "specific task"}}],
  "reasoning": "brief explanation"
}}"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    
    try:
        response = await chain.ainvoke({"query": state['question'], "context": context})
        json_str = response.content.strip()
        start = json_str.find('{')
        end = json_str.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found")
        json_obj = json.loads(json_str[start:end])
        plan = CoordinatorPlan(**json_obj)
        logger.info(f"Plan created: {len(plan.plan)} steps")
        return {
            "coordinator_analysis": {"plan": [s.dict() for s in plan.plan], "reasoning": plan.reasoning},
            "current_step": 0,
            "execution_results": {}
        }
    except Exception as e:
        logger.warning(f"Coordinator fallback: {e}")
        # Fallback routing
        fallback = "ADVISOR"
        if state.get("pdf_path") and any(w in state['question'].lower() for w in ['document', 'pdf', 'paper']):
            fallback = "RESEARCHER"
        elif any(w in state['question'].lower() for w in ['notion', 'note', 'calendar']):
            fallback = "NOTIONWRITER"
        elif any(w in state['question'].lower() for w in ['plan', 'todo', 'schedule']):
            fallback = "PLANNER"
        elif any(w in state['question'].lower() for w in ['search', 'web', 'latest', 'news']):
            fallback = "WEB_RESEARCHER"
        return {
            "coordinator_analysis": {"plan": [{"agent": fallback, "query": state['question']}], "reasoning": f"Fallback to {fallback}"},
            "current_step": 0,
            "execution_results": {}
        }

async def execute_agent_step(state: AgentState, user_profile: Dict) -> Dict:
    plan = state.get("coordinator_analysis", {}).get("plan", [])
    current_step = state.get("current_step", 0)
    
    if current_step >= len(plan):
        return {"current_step": current_step}
    
    step = plan[current_step]
    agent_name = step["agent"]
    agent_query = step["query"]
    logger.info(f"Step {current_step + 1}/{len(plan)}: {agent_name}")
    
    results = state.get("execution_results", {})
    
    if agent_name == "RESEARCHER":
        output = await researcher_rag_node({**state, "question": agent_query})
        results["researcher"] = output.get("researcher_output", "")
        return {"researcher_output": results["researcher"], "execution_results": results, "current_step": current_step + 1}
    
    elif agent_name == "WEB_RESEARCHER":
        web_result = await web_app.ainvoke({"question": agent_query, "chat_history": state.get("chat_history", [])})
        if web_result.get("intermediate_steps"):
            output = web_result["intermediate_steps"][-1][1]
        elif isinstance(web_result.get("agent_outcome"), dict):
            output = web_result["agent_outcome"].get("output", "Web search done.")
        else:
            output = "Web research done."
        results["web_researcher"] = output
        return {"web_researcher_output": output, "execution_results": results, "current_step": current_step + 1}
    
    elif agent_name == "NOTIONWRITER":
        notion_result = await notion_app.ainvoke({"question": agent_query, "chat_history": state.get("chat_history", [])})
        if notion_result.get("intermediate_steps"):
            output = notion_result["intermediate_steps"][-1][1]
        elif hasattr(notion_result.get("agent_outcome"), "return_values"):
            output = notion_result["agent_outcome"].return_values.get("output", "Notion done.")
        else:
            output = "Notion operation done."
        results["notion"] = output
        return {"notewriter_output": output, "execution_results": results, "current_step": current_step + 1}
    
    elif agent_name == "PLANNER":
        planner_result = await planner_app.ainvoke({"question": agent_query, "chat_history": state.get("chat_history", [])})
        if planner_result.get("intermediate_steps"):
            output = planner_result["intermediate_steps"][-1][1]
        elif hasattr(planner_result.get("agent_outcome"), "return_values"):
            output = planner_result["agent_outcome"].return_values.get("output", "Planning done.")
        else:
            output = "Planning done."
        results["planner"] = output
        return {"planner_output": output, "execution_results": results, "current_step": current_step + 1}
    
    elif agent_name == "ADVISOR":
        output = await advisor_analyze_node({**state, "question": agent_query}, user_profile)
        results["advisor"] = output.get("advisor_output", "")
        return {"advisor_output": results["advisor"], "execution_results": results, "current_step": current_step + 1}
    
    else:
        results["error"] = f"Unknown agent: {agent_name}"
        return {"execution_results": results, "current_step": current_step + 1}

async def generate_final_response(state: AgentState) -> Dict:
    logger.info("Synthesizing final response...")
    execution_results = state.get("execution_results", {})
    
    # Single agent direct output
    if len(execution_results) == 1:
        single_output = list(execution_results.values())[0]
        if single_output and "error" not in single_output.lower():
            return {"answer": single_output}
    
    # Synthesize multiple outputs
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Synthesize information from agents into a coherent response."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "QUERY: {query}\n\nAGENT OUTPUTS:\n{outputs}\n\nSynthesize:")
    ])
    
    outputs_text = "\n\n".join([f"**{agent.upper()}**:\n{output}" for agent, output in execution_results.items()])
    chain = prompt | llm
    
    try:
        response = await chain.ainvoke({"query": state.get('question', ''), "outputs": outputs_text, "chat_history": state.get('chat_history', [])})
        return {"answer": response.content.strip()}
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        fallback = "\n\n".join(execution_results.values())
        return {"answer": fallback or "Error generating response."}

async def speech_synthesis_node(state: AgentState) -> Dict:
    logger.info("Generating speech...")
    answer = state.get("answer", "")
    if not answer or len(answer) < 100:
        return {"audio_output_path": speak_text(answer)}
    
    # Summarize for speech
    prompt = ChatPromptTemplate.from_template("Summarize concisely for speech (2-3 sentences): {text}")
    try:
        chain = prompt | llm
        summary = await chain.ainvoke({"text": answer})
        return {"audio_output_path": speak_text(summary.content.strip())}
    except Exception as e:
        logger.error(f"Speech error: {e}")
        return {"audio_output_path": None}

def should_continue_execution(state: AgentState) -> str:
    plan = state.get("coordinator_analysis", {}).get("plan", [])
    current_step = state.get("current_step", 0)
    return "execute_step" if current_step < len(plan) else "synthesize"

def create_main_graph(user_profile: Dict = None):
    async def execute_step_wrapper(state: AgentState) -> Dict:
        return await execute_agent_step(state, user_profile or {})
    
    async def coordinator_wrapper(state: AgentState) -> Dict:
        return await coordinator_node(state)
    
    async def synthesize_wrapper(state: AgentState) -> Dict:
        return await generate_final_response(state)
    
    async def speech_wrapper(state: AgentState) -> Dict:
        return await speech_synthesis_node(state)
    
    graph = StateGraph(AgentState)
    graph.add_node("coordinator", coordinator_wrapper)
    graph.add_node("execute_step", execute_step_wrapper)
    graph.add_node("synthesize", synthesize_wrapper)
    graph.add_node("speech", speech_wrapper)
    graph.set_entry_point("coordinator")
    graph.add_conditional_edges("coordinator", lambda state: "execute_step", {"execute_step": "execute_step"})
    graph.add_conditional_edges("execute_step", should_continue_execution, {"execute_step": "execute_step", "synthesize": "synthesize"})
    graph.add_edge("synthesize", "speech")
    graph.add_edge("speech", END)
    return graph.compile()

# ==============================================================================
# STREAMLIT UI
# ==============================================================================

st.set_page_config(page_title="ATLAS", page_icon="📚", layout="wide")

def initialize_session_state():
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'pdf_path' not in st.session_state:
        st.session_state.pdf_path = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {"academic_level": "", "major": "", "interests": "", "description": ""}

async def process_query(query: str):
    try:
        user_profile = st.session_state.user_profile
        inputs = {
            "question": query,
            "pdf_path": st.session_state.get("pdf_path"),
            "user_profile": user_profile,
            "chat_history": st.session_state.memory.load_memory_variables({})["chat_history"]
        }
        
        with st.spinner("🤖 Initializing..."):
            graph = create_main_graph(user_profile)
        
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            answer_placeholder = st.empty()
            final_answer, audio_path = "", None
            
            friendly_names = {
                "coordinator": "🎯 Analyzing",
                "execute_step": "⚙️ Executing",
                "synthesize": "✨ Synthesizing",
                "speech": "🔊 Audio"
            }
            
            async for event in graph.astream(inputs):
                for node_name, output in event.items():
                    status_placeholder.markdown(f"*{friendly_names.get(node_name, node_name)}...*")
                    if "answer" in output:
                        final_answer = output["answer"]
                        answer_placeholder.markdown(final_answer)
                    if "audio_output_path" in output:
                        audio_path = output["audio_output_path"]
            
            status_placeholder.empty()
            if not final_answer:
                final_answer = "I couldn't generate a response. Please try again."
                answer_placeholder.markdown(final_answer)
        
        if audio_path:
            play_audio(audio_path)
        
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        st.session_state.memory.save_context({"input": query}, {"output": final_answer})
        
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        with st.chat_message("assistant"):
            st.error("❌ An error occurred. Please try again.")

def main():
    st.title("📚 ATLAS")
    st.caption("Academic Task & Learning Agent System")
    
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # User Profile
        with st.expander("👤 User Profile", expanded=False):
            profile = st.session_state.user_profile
            academic_level = st.selectbox("Academic Level", ["", "High School", "Undergraduate", "Graduate", "PhD", "Professional"], 
                                         index=0 if not profile.get("academic_level") else ["", "High School", "Undergraduate", "Graduate", "PhD", "Professional"].index(profile.get("academic_level")))
            major = st.text_input("Major", value=profile.get("major", ""))
            interests = st.text_input("Interests", value=profile.get("interests", ""))
            description = st.text_area("Additional Info", value=profile.get("description", ""), height=100)
            
            if st.button("💾 Save Profile"):
                st.session_state.user_profile = {"academic_level": academic_level, "major": major, "interests": interests, "description": description}
                st.success("✅ Profile saved!")
        
        st.divider()
        st.header("📄 Document")
        
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file:
            if st.session_state.get("uploaded_file_name") != uploaded_file.name:
                if st.session_state.get("pdf_path"):
                    try:
                        os.remove(st.session_state.pdf_path)
                    except:
                        pass
                st.session_state.pdf_path = None
                st.session_state.uploaded_file_name = None
                get_or_create_vector_store.clear()
            
            if not st.session_state.get("pdf_path"):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        st.session_state.pdf_path = tmp.name
                        st.session_state.uploaded_file_name = uploaded_file.name
                    st.success(f"✅ {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        if st.session_state.get("pdf_path"):
            st.info(f"📄 {st.session_state.uploaded_file_name}")
            if st.button("🗑️ Clear"):
                try:
                    os.remove(st.session_state.pdf_path)
                except:
                    pass
                st.session_state.pdf_path = None
                st.session_state.uploaded_file_name = None
                get_or_create_vector_store.clear()
                st.rerun()
        
        if st.button("🔄 Clear Chat"):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.rerun()
    
    # Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Welcome message
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            welcome = """👋 Welcome to ATLAS!

I can help with:
- 📄 PDF Analysis
- 🌐 Web Research  
- 📝 Notion Notes
- 📅 Task Planning
- 🎓 Academic Advice

How can I assist you?"""
            st.markdown(welcome)
        st.session_state.messages.append({"role": "assistant", "content": welcome})
    
    # Input
    col1, col2 = st.columns([6, 1])
    with col2:
        use_voice = st.button("🎤", help="Voice input")
    
    query = st.chat_input("Ask ATLAS...")
    
    if use_voice:
        if record_audio():
            with st.spinner("📝 Transcribing..."):
                transcript = transcribe_audio("temp_recording.wav")
            if transcript:
                query = transcript
                st.success(f"✅ {transcript}")
            else:
                st.warning("❌ Transcription failed")
            if os.path.exists("temp_recording.wav"):
                try:
                    os.remove("temp_recording.wav")
                except:
                    pass
        else:
            st.warning("⚠️ No speech detected")
    
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        asyncio.run(process_query(query))

if __name__ == "__main__":
    main()