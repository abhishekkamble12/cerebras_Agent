# cerebras_Agent



📚 ATLAS: Academic Task & Learning Agent System
ATLAS is an intelligent academic assistant designed to streamline your learning and research workflow. Powered by a multi-agent system built with LangChain and LangGraph, ATLAS integrates powerful AI capabilities into a conversational interface, accessible via text or voice.

Whether you need to analyze research papers, browse the web for the latest information, manage your notes in Notion, or plan your study schedule, ATLAS acts as your all-in-one academic partner.

(This is a placeholder image. Replace with a screenshot of your Streamlit app)

✨ Core Features
ATLAS is built on a sophisticated multi-agent architecture that intelligently routes tasks to the right specialist agent.

🗣️ Voice Interface: Interact with ATLAS naturally using your voice. It listens, transcribes, processes your request, and speaks back the answer.

📄 PDF Document Analysis (RAG): Upload a PDF (like a research paper or textbook chapter) and ask questions directly about its content. ATLAS uses a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers.

🌐 Web Researcher: Ask about current events, the latest research, or any topic that requires up-to-date information. The Web Researcher agent uses Tavily Search to browse the web and can scrape websites for detailed content.

📝 Notion Integration: Seamlessly manage your academic life in Notion.

Create & Update Notes: Add new notes or append information to existing pages.

Schedule Events: Add lectures, deadlines, or study sessions to your Notion calendar with specific times and dates.

Search & Retrieve: Find information stored in your Notion databases.

Delete Pages: Archive old or irrelevant notes.

📅 Task Planner & Scheduler: Organize your academic tasks efficiently.

Break down complex assignments into manageable steps.

Create to-do lists and study schedules.

Prioritize tasks based on urgency and importance.

🧑‍🏫 Personalized Academic Advisor: Get tailored academic advice. By setting a simple user profile (e.g., "3rd-year Physics major"), the Advisor agent provides guidance that is relevant to your specific context.

🧠 Intelligent Orchestration: A central "coordinator" agent analyzes your request and creates a sequential execution plan, chaining multiple agents together to handle complex, multi-step queries (e.g., "Search the web for new papers on quantum computing and add a summary to my Notion").

🛠️ Tech Stack
Frameworks: LangChain, LangGraph, Streamlit

LLM Providers: Cerebras (primary), OpenAI (fallback)

Vector Store: FAISS with Hugging Face embeddings for RAG

Tools & Services:

Web Search: Tavily API

Note-Taking: Notion API

Voice I/O: Deepgram (STT), gTTS (TTS), Silero-VAD

🚀 Getting Started
Follow these steps to set up and run ATLAS on your local machine.

1. Prerequisites
Python 3.10+

An .env file with the necessary API keys (see env.example).

2. Installation
Clone the repository:

Bash

git clone <your-repository-url>
cd <your-repository-directory>
Create a virtual environment and install dependencies:

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
(Note: A requirements.txt file should be created from the script's imports)

3. Configuration
Create an environment file:
Copy the env.example file to a new file named .env.

Bash

cp env.example .env
Add your API keys to .env:
You will need to sign up for accounts with the following services to get your keys:

CEREBRAS_API_KEY or OPENAI_API_KEY (at least one is required)

DEEPGRAM_API_KEY

TAVILY_API_KEY

NOTION_TOKEN, DATABASE_ID (for notes), DATABASE_ID1 (for calendar)

For Notion, follow the official guide to create an integration and share your databases with it.

4. Running the Application
Start the Streamlit application:

Bash

streamlit run app.py
Open your browser:
Navigate to the local URL provided by Streamlit (usually http://localhost:8501).

💬 How to Use ATLAS
(Optional) Set Your Profile: In the sidebar, enter your academic level, major, and interests for personalized advice.

(Optional) Upload a PDF: Use the file uploader in the sidebar to load a document for analysis.

Ask Away!: Use the chat input at the bottom to type your question, or click the "🎤 Ask with Voice" button to speak.

Example Prompts:
PDF Question: "Summarize the methodology section of the uploaded document."

Web Research: "Search for the latest breakthroughs in CRISPR technology."

Notion Task: "Add a note to Notion titled 'Thesis Ideas' with the content 'Explore the impact of AI on renewable energy grids'."

Planning: "Help me create a study schedule for my final exams in Calculus, History, and Chemistry for the next 5 days."

Complex Query: "Search the web for the top 3 machine learning libraries, then create a new note in Notion with a summary of each."
<img width="978" height="842" alt="image" src="https://github.com/user-attachments/assets/970b46ba-d4b5-4a92-908a-21232315f46b" />
<img width="978" height="842" alt="image" src="https://github.com/user-attachments/assets/3186e5c3-693e-457b-9d6f-903217ceafa8" />

