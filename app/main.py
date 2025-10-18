# app/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import Any, List, Optional
from pathlib import Path
import re

from .models import Gem, GemCreate, GemUpdate, ChatRequest, ChatResponse, Message
from . import store
from .tools import list_tools, run_tool
from .llm import chat as llm_chat
from . import kb
from fastapi.responses import HTMLResponse

app = FastAPI(title="Gems Agent API", version="0.2.0")

TOOLS_INSTRUCTION = (
    "You have access to the following tools: {tools}.\n"
    "When you want to call a tool, reply with ONLY this JSON (no extra text):\n"
    "{{\"tool\":\"<tool_name>\",\"input\":\"<text>\"}}\n"
    "After the tool result is provided, produce a concise final answer for the user.\n"
)

@app.get("/health")
def health():
    return {"status": "ok", "tools": list_tools()}

# ---------- Templates ----------
@app.get("/templates")
def get_templates():
    return {
        "templates": [
            {
                "id": "travel_assistant",
                "name": "Travel Assistant",
                "description": "Helps with travel planning, destination research, and trip organization",
                "system_prompt": """You are a helpful travel assistant. Your role is to:
- Help users plan their trips and vacations
- Provide detailed information about destinations, attractions, and activities
- Suggest itineraries based on user preferences and budget
- Answer travel-related questions about visas, weather, culture, and safety
- Give practical travel tips and recommendations
- Help with booking suggestions and travel logistics

Always be friendly, informative, and considerate of the user's budget, time constraints, and travel preferences. Provide specific, actionable advice and include relevant details like costs, timing, and alternatives when possible.""",
                "tools": ["web_search", "calculator"],
                "model": None
            },
            {
                "id": "code_reviewer",
                "name": "Code Reviewer",
                "description": "Reviews code, suggests improvements, and helps with programming best practices",
                "system_prompt": """You are an expert code reviewer and programming mentor. Your role is to:
- Review code for bugs, performance issues, and security vulnerabilities
- Suggest improvements for code quality, readability, and maintainability
- Explain programming concepts and best practices
- Help with debugging and troubleshooting
- Provide guidance on architecture and design patterns
- Suggest refactoring opportunities

Always be constructive in your feedback, explain the reasoning behind suggestions, and provide code examples when helpful. Focus on teaching and helping developers improve their skills.""",
                "tools": ["web_search"],
                "model": None
            },
            {
                "id": "research_assistant",
                "name": "Research Assistant",
                "description": "Conducts research, summarizes information, and helps with academic work",
                "system_prompt": """You are a research assistant specializing in academic and professional research. Your role is to:
- Help users find and analyze relevant information on various topics
- Summarize complex research papers and articles
- Suggest research methodologies and approaches
- Help structure research questions and hypotheses
- Provide guidance on academic writing and citation
- Assist with literature reviews and data analysis

Always maintain academic integrity, cite sources properly, and provide balanced, evidence-based perspectives. Help users develop critical thinking skills and research capabilities.""",
                "tools": ["web_search", "kb_search"],
                "model": None
            },
            {
                "id": "customer_support",
                "name": "Customer Support Agent",
                "description": "Handles customer inquiries, provides support, and resolves issues",
                "system_prompt": """You are a professional customer support agent. Your role is to:
- Help customers with their questions and concerns
- Provide clear, helpful, and accurate information
- Resolve issues efficiently and professionally
- Escalate complex problems when necessary
- Maintain a friendly and empathetic tone
- Follow company policies and procedures

Always be patient, understanding, and solution-oriented. Focus on customer satisfaction and building positive relationships. If you don't know something, be honest and offer to find the right person who can help.""",
                "tools": ["kb_search"],
                "model": None
            },
            {
                "id": "content_writer",
                "name": "Content Writer",
                "description": "Creates engaging content, writes articles, and helps with marketing materials",
                "system_prompt": """You are a professional content writer and marketing specialist. Your role is to:
- Create engaging, high-quality content for various platforms
- Write articles, blog posts, social media content, and marketing copy
- Adapt writing style to different audiences and purposes
- Help with content strategy and planning
- Suggest improvements for existing content
- Ensure content is SEO-friendly and brand-consistent

Always write with the target audience in mind, maintain a consistent brand voice, and create content that provides real value to readers. Focus on clarity, engagement, and actionable insights.""",
                "tools": ["web_search"],
                "model": None
            },
            {
                "id": "data_analyst",
                "name": "Data Analyst",
                "description": "Analyzes data, creates visualizations, and provides insights",
                "system_prompt": """You are a data analyst and business intelligence specialist. Your role is to:
- Help users analyze data and extract meaningful insights
- Suggest appropriate statistical methods and analytical approaches
- Help interpret results and identify trends and patterns
- Provide guidance on data visualization and reporting
- Assist with data cleaning and preparation
- Explain complex analytical concepts in simple terms

Always be thorough in your analysis, explain your reasoning clearly, and provide actionable insights that help users make informed decisions. Focus on practical applications and business value.""",
                "tools": ["calculator", "web_search"],
                "model": None
            }
        ]
    }

# ---------- Gems CRUD ----------
@app.get("/gems")
def list_gems():
    return [g.model_dump() for g in store.load_all()]

@app.get("/gems/{gem_id}")
def get_gem(gem_id: str):
    gem = store.get_gem(gem_id)
    if not gem:
        raise HTTPException(404, "Gem not found")
    return gem.model_dump()

@app.post("/gems")
def create_gem(body: GemCreate):
    new = Gem(
        id=store.new_id(),
        name=body.name,
        system_prompt=body.system_prompt,
        tools=body.tools or [],
        temperature=body.temperature or 0.2,
        model=body.model
    )
    store.add_gem(new)
    return new.model_dump()

@app.put("/gems/{gem_id}")
def update_gem(gem_id: str, patch: GemUpdate):
    updated = store.update_gem(gem_id, patch.model_dump())
    if not updated:
        raise HTTPException(404, "Gem not found")
    return updated.model_dump()

@app.delete("/gems/{gem_id}")
def remove_gem(gem_id: str):
    ok = store.delete_gem(gem_id)
    if not ok:
        raise HTTPException(404, "Gem not found")
    return {"deleted": True}

# ---------- KB/Files ----------
@app.post("/gems/{gem_id}/files")
async def upload_files(gem_id: str, files: List[UploadFile] = File(...)):
    gem = store.get_gem(gem_id)
    if not gem:
        raise HTTPException(404, "Gem not found")

    if not files:
        raise HTTPException(400, "No files provided")

    tmp_paths: List[Path] = []
    try:
        for f in files:
            # Проверяем, что filename не None
            filename = f.filename or "unnamed_file"
            p = Path("/tmp") / filename
            
            # Читаем содержимое файла
            content = await f.read()
            if not content:
                continue  # Пропускаем пустые файлы
                
            p.write_bytes(content)
        tmp_paths.append(p)

        if not tmp_paths:
            raise HTTPException(400, "No valid files to process")

        info = kb.ingest_files(gem_id, tmp_paths)

        # если kb_search ещё не в инструментах — добавим
        if "kb_search" not in (gem.tools or []):
            store.update_gem(gem_id, {"tools": (gem.tools or []) + ["kb_search"]})

        return info

    except Exception as e:
        raise HTTPException(500, f"Error processing files: {str(e)}")

    finally:
        # удаляем времянки
        for p in tmp_paths:
            try:
                p.unlink()
            except Exception:
                pass

@app.get("/gems/{gem_id}/files")
def list_agent_files(gem_id: str):
    if not store.get_gem(gem_id):
        raise HTTPException(404, "Gem not found")
    return {"files": kb.list_files(gem_id)}

@app.get("/gems/{gem_id}/kb/status")
def kb_status(gem_id: str):
    return kb.status(gem_id)

# ---------- Chat ----------
@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    gem = store.get_gem(body.gem_id)
    if not gem:
        raise HTTPException(404, "Gem not found")

    # 1) system + инструменты
    sys = gem.system_prompt
    if body.tools_mode == "auto" and gem.tools:
        sys += "\n\n" + TOOLS_INSTRUCTION.format(tools=', '.join(gem.tools))

    convo = [{"role": "system", "content": sys}]
    for m in body.messages:
        convo.append({"role": m.role, "content": m.content})

    # 2) RAG-контекст на основе запроса пользователя
    last_user = next((m.content for m in reversed(body.messages) if m.role == "user"), "")
    if last_user and kb.has_index(gem.id):
        snips = kb.query(gem.id, last_user, k=4)
        ctx = kb.build_context(snips)
        if ctx:
            # даём как system, чтобы LLM опирался на факты
            convo.append({"role": "system", "content": ctx})

    # 3) первый ход модели
    first = llm_chat(convo, temperature=gem.temperature, model_override=gem.model)

    # 4) авто-вызов инструмента по JSON {"tool":"...","input":"..."}
    used_tool: Optional[str] = None
    tool_input: Optional[str] = None
    if body.tools_mode == "auto" and gem.tools:
        # ищем JSON с экранированными кавычками (часто так отвечает LLM)
        tool_match = re.search(
            r'\{\s*\\"tool\\"\s*:\s*\\"([^\\"]+)\\"\s*,\s*\\"input\\"\s*:\s*\\"([\s\S]*?)\\"\s*\}',
            first
        )
        if tool_match:
            tname = tool_match.group(1).strip()
            tinp = tool_match.group(2).strip()
            if tname in gem.tools:
                used_tool = tname
                tool_input = tinp
                tool_result = run_tool(tname, tinp, gem_id=gem.id)

                # feed back: что сказал ассистент и что вернул инструмент
                convo.append({"role": "assistant", "content": first})
                convo.append({"role": "tool", "content": f"Tool {tname} result:\n{tool_result}"})

                final = llm_chat(convo, temperature=gem.temperature, model_override=gem.model)
                return ChatResponse(content=final, used_tool=used_tool, tool_input=tool_input)

    # 5) без инструмента — сразу отдаём ответ
    return ChatResponse(content=first, used_tool=used_tool, tool_input=tool_input)

# --------- (необязательно) простая страница конструктора ---------
@app.get("/manage", response_class=HTMLResponse)
def manage():
    return """
<!doctype html><html><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Gems Creator - AI Agent Builder</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
  .file-drop-zone { 
    border: 2px dashed #cbd5e1; 
    transition: all 0.3s ease;
  }
  .file-drop-zone.dragover { 
    border-color: #3b82f6; 
    background-color: #eff6ff; 
  }
  .gem-card { 
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .gem-card:hover { 
    transform: translateY(-2px); 
    box-shadow: 0 10px 25px rgba(0,0,0,0.1); 
  }
</style>
</head><body class="bg-gradient-to-br from-slate-50 to-slate-100 min-h-screen">
<div class="max-w-6xl mx-auto p-6 space-y-8">
  <!-- Header -->
  <div class="text-center mb-8">
    <h1 class="text-4xl font-bold text-gray-800 mb-2">🤖 Gems Creator</h1>
    <p class="text-gray-600">Create AI agents with custom instructions and knowledge base</p>
  </div>

  <!-- Templates Section -->
  <section class="bg-white rounded-2xl shadow-lg p-6 gem-card">
    <div class="flex items-center mb-6">
      <div class="w-10 h-10 bg-yellow-100 rounded-lg flex items-center justify-center mr-3">
        <span class="text-yellow-600 text-xl">🚀</span>
      </div>
      <h2 class="text-xl font-semibold text-gray-800">Quick Start Templates</h2>
    </div>
    
    <div id="templatesGrid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      <!-- Templates will be loaded here -->
    </div>
  </section>

  <!-- Main Content Grid -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
    
    <!-- Create/Edit Gem Section -->
    <section class="bg-white rounded-2xl shadow-lg p-6 gem-card">
      <div class="flex items-center mb-6">
        <div class="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center mr-3">
          <span class="text-indigo-600 text-xl">⚙️</span>
        </div>
        <h2 class="text-xl font-semibold text-gray-800">Agent Configuration</h2>
      </div>
      
      <div class="space-y-4">
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">Agent Name</label>
          <input id="name" class="w-full border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-indigo-500 focus:border-transparent" placeholder="e.g., Travel Assistant, Code Reviewer, Research Helper"/>
        </div>
        
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">System Instructions</label>
          <div class="relative">
            <textarea id="sys" rows="16" class="w-full border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none" placeholder="Write detailed instructions for your AI agent...

Example:
You are a helpful travel assistant. Your role is to:
- Help users plan their trips
- Provide information about destinations
- Suggest activities and attractions
- Answer travel-related questions
- Give practical travel tips

Always be friendly, informative, and considerate of the user's budget and preferences."></textarea>
            <div class="absolute bottom-2 right-2 text-xs text-gray-400">
              <span id="charCount">0</span> characters
            </div>
          </div>
        </div>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">Tools</label>
            <input id="tools" class="w-full border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-indigo-500 focus:border-transparent" placeholder="web_search, calculator, kb_search"/>
            <p class="text-xs text-gray-500 mt-1">Comma-separated list of available tools</p>
          </div>
          
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">Model</label>
            <select id="model" class="w-full border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-indigo-500 focus:border-transparent">
              <option value="">Default (llama3.1:8b)</option>
              <option value="gemini-2.0-flash">Gemini 2.0 Flash (FREE)</option>
              <option value="gpt-4o-mini">GPT-4o-mini</option>
              <option value="gpt-4">GPT-4</option>
              <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
              <option value="claude-3-sonnet">Claude 3 Sonnet</option>
              <option value="llama3.1:70b">Llama 3.1 70B</option>
            </select>
          </div>
        </div>
        
        <div class="flex items-center justify-between pt-4">
          <div class="flex items-center">
            <input type="checkbox" id="autoTools" class="mr-2" checked>
            <label for="autoTools" class="text-sm text-gray-600">Auto-enable tools</label>
          </div>
          <button id="create" class="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors font-medium">
            💾 Save Agent
          </button>
        </div>
        
        <div id="createOut" class="text-sm mt-3 p-3 rounded-lg hidden"></div>
      </div>
  </section>

    <!-- File Upload Section -->
    <section class="bg-white rounded-2xl shadow-lg p-6 gem-card">
      <div class="flex items-center mb-6">
        <div class="w-10 h-10 bg-emerald-100 rounded-lg flex items-center justify-center mr-3">
          <span class="text-emerald-600 text-xl">📁</span>
        </div>
        <h2 class="text-xl font-semibold text-gray-800">Knowledge Base</h2>
      </div>
      
      <div class="space-y-4">
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">Select Agent</label>
          <select id="agent" class="w-full border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-emerald-500 focus:border-transparent">
            <option value="">Choose an agent...</option>
          </select>
        </div>
        
        <div class="file-drop-zone rounded-lg p-8 text-center cursor-pointer" id="dropZone">
          <div class="space-y-4">
            <div class="text-4xl">📤</div>
            <div>
              <p class="text-lg font-medium text-gray-700">Drop files here or click to browse</p>
              <p class="text-sm text-gray-500">Supports PDF, TXT, DOC, images and more</p>
            </div>
            <input id="files" type="file" multiple class="hidden" accept=".pdf,.txt,.doc,.docx,.jpg,.jpeg,.png,.gif,.md">
            <button class="px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors">
              Choose Files
            </button>
          </div>
        </div>
        
        <div id="fileList" class="space-y-2"></div>
        
        <div class="flex items-center justify-between pt-4">
          <div class="text-sm text-gray-600">
            <span id="fileCount">0</span> files selected
          </div>
          <button id="upload" class="px-6 py-3 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors font-medium disabled:opacity-50" disabled>
            🚀 Upload Files
          </button>
        </div>
        
        <div id="upOut" class="text-sm mt-3 p-3 rounded-lg hidden"></div>
        <div id="kbStat" class="text-xs mt-2 text-gray-600"></div>
      </div>
    </section>
  </div>

  <!-- Agent Preview & Testing Section -->
  <section class="bg-white rounded-2xl shadow-lg p-6 gem-card">
    <div class="flex items-center mb-6">
      <div class="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center mr-3">
        <span class="text-purple-600 text-xl">💬</span>
      </div>
      <h2 class="text-xl font-semibold text-gray-800">Test Your Agent</h2>
    </div>
    
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-2">Select Agent to Test</label>
        <select id="testAgent" class="w-full border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-purple-500 focus:border-transparent">
          <option value="">Choose an agent...</option>
        </select>
      </div>
      
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-2">Your Message</label>
        <div class="flex space-x-2">
          <input id="testMessage" class="flex-1 border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-purple-500 focus:border-transparent" placeholder="Ask your agent something..."/>
          <button id="sendTest" class="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors font-medium" disabled>
            Send
          </button>
        </div>
      </div>
    </div>
    
    <div id="testResponse" class="mt-4 p-4 bg-gray-50 rounded-lg min-h-[100px] hidden">
      <div class="text-sm text-gray-600 mb-2">Agent Response:</div>
      <div id="testContent" class="text-gray-800"></div>
    </div>
  </section>

  <!-- Current Agents List -->
  <section class="bg-white rounded-2xl shadow-lg p-6 gem-card">
    <div class="flex items-center justify-between mb-6">
      <div class="flex items-center">
        <div class="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center mr-3">
          <span class="text-blue-600 text-xl">📋</span>
        </div>
        <h2 class="text-xl font-semibold text-gray-800">Your Agents</h2>
      </div>
      <button id="refreshAgents" class="px-4 py-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors">
        🔄 Refresh
      </button>
    </div>
    
    <div id="agentsList" class="space-y-4">
      <div class="text-center text-gray-500 py-8">
        <div class="text-4xl mb-2">🤖</div>
        <p>No agents created yet. Create your first agent above!</p>
      </div>
    </div>
  </section>
</div>

<script>
const API = "";

// Global state
let selectedFiles = [];
let agents = [];
let templates = [];

// Utility functions
function showNotification(elementId, message, type = 'info') {
  const element = document.getElementById(elementId);
  element.textContent = message;
  element.className = `text-sm mt-3 p-3 rounded-lg ${type === 'error' ? 'bg-red-100 text-red-700' : type === 'success' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'}`;
  element.classList.remove('hidden');
  setTimeout(() => element.classList.add('hidden'), 5000);
}

function updateCharCount() {
  const textarea = document.getElementById('sys');
  const charCount = document.getElementById('charCount');
  charCount.textContent = textarea.value.length;
}

function updateFileCount() {
  const fileCount = document.getElementById('fileCount');
  fileCount.textContent = selectedFiles.length;
  document.getElementById('upload').disabled = selectedFiles.length === 0 || !document.getElementById('agent').value;
}

// Load templates
async function loadTemplates() {
  try {
    const response = await fetch(API + "/templates");
    const data = await response.json();
    templates = data.templates;
    displayTemplates();
  } catch (error) {
    console.error('Error loading templates:', error);
  }
}

function displayTemplates() {
  const templatesGrid = document.getElementById('templatesGrid');
  
  templatesGrid.innerHTML = templates.map(template => `
    <div class="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors cursor-pointer" onclick="useTemplate('${template.id}')">
      <div class="flex items-start space-x-3">
        <div class="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
          <span class="text-blue-600 text-sm">🤖</span>
        </div>
        <div class="flex-1 min-w-0">
          <h3 class="font-semibold text-gray-800 text-sm">${template.name}</h3>
          <p class="text-xs text-gray-600 mt-1">${template.description}</p>
          <div class="flex items-center space-x-2 mt-2">
            <span class="text-xs text-gray-500">Tools: ${template.tools.length}</span>
            <span class="text-xs text-blue-600">Click to use</span>
          </div>
        </div>
      </div>
    </div>
  `).join('');
}

function useTemplate(templateId) {
  const template = templates.find(t => t.id === templateId);
  if (!template) return;
  
  // Fill the form with template data
  document.getElementById('name').value = template.name;
  document.getElementById('sys').value = template.system_prompt;
  document.getElementById('tools').value = template.tools.join(', ');
  document.getElementById('model').value = template.model || '';
  updateCharCount();
  
  // Scroll to the form
  document.querySelector('section:nth-of-type(2)').scrollIntoView({ behavior: 'smooth' });
  
  showNotification('createOut', `Template "${template.name}" loaded!`, 'success');
}

// Load and display agents
async function loadAgents() {
  try {
    const response = await fetch(API + "/gems");
    agents = await response.json();
    
    // Update agent selectors
    const agentSelect = document.getElementById("agent");
    const testAgentSelect = document.getElementById("testAgent");
    
    [agentSelect, testAgentSelect].forEach(select => {
      select.innerHTML = '<option value="">Choose an agent...</option>';
      agents.forEach(agent => {
        const option = document.createElement("option");
        option.value = agent.id;
        option.textContent = agent.name;
        select.appendChild(option);
      });
    });
    
    // Update agents list display
    displayAgentsList();
    
  } catch (error) {
    console.error('Error loading agents:', error);
    showNotification('createOut', 'Error loading agents', 'error');
  }
}

function displayAgentsList() {
  const agentsList = document.getElementById('agentsList');
  
  if (agents.length === 0) {
    agentsList.innerHTML = `
      <div class="text-center text-gray-500 py-8">
        <div class="text-4xl mb-2">🤖</div>
        <p>No agents created yet. Create your first agent above!</p>
      </div>
    `;
    return;
  }
  
  agentsList.innerHTML = agents.map(agent => `
    <div class="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors">
      <div class="flex items-center justify-between">
        <div class="flex-1">
          <h3 class="font-semibold text-gray-800">${agent.name}</h3>
          <p class="text-sm text-gray-600 mt-1">${agent.system_prompt.substring(0, 100)}${agent.system_prompt.length > 100 ? '...' : ''}</p>
          <div class="flex items-center space-x-4 mt-2 text-xs text-gray-500">
            <span>Tools: ${agent.tools?.length || 0}</span>
            <span>Model: ${agent.model || 'Default'}</span>
            <span>Temp: ${agent.temperature}</span>
          </div>
        </div>
        <div class="flex space-x-2">
          <button onclick="editAgent('${agent.id}')" class="px-3 py-1 text-blue-600 hover:bg-blue-50 rounded text-sm">
            ✏️ Edit
          </button>
          <button onclick="deleteAgent('${agent.id}')" class="px-3 py-1 text-red-600 hover:bg-red-50 rounded text-sm">
            🗑️ Delete
          </button>
        </div>
      </div>
    </div>
  `).join('');
}

// Agent management
async function createAgent() {
  const name = document.getElementById('name').value.trim();
  const systemPrompt = document.getElementById('sys').value.trim();
  const tools = document.getElementById('tools').value.split(',').map(s => s.trim()).filter(Boolean);
  const model = document.getElementById('model').value.trim() || null;
  
  if (!name || !systemPrompt) {
    showNotification('createOut', 'Please fill in name and system instructions', 'error');
    return;
  }
  
  try {
    const body = { name, system_prompt: systemPrompt, tools, model };
    const response = await fetch(API + "/gems", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    
    if (response.ok) {
      showNotification('createOut', 'Agent created successfully!', 'success');
      // Clear form
      document.getElementById('name').value = '';
      document.getElementById('sys').value = '';
      document.getElementById('tools').value = '';
      document.getElementById('model').value = '';
      updateCharCount();
      loadAgents();
    } else {
      const error = await response.text();
      showNotification('createOut', `Error: ${error}`, 'error');
    }
  } catch (error) {
    showNotification('createOut', `Error: ${error.message}`, 'error');
  }
}

async function deleteAgent(agentId) {
  if (!confirm('Are you sure you want to delete this agent?')) return;
  
  try {
    const response = await fetch(API + `/gems/${agentId}`, { method: "DELETE" });
    if (response.ok) {
      showNotification('createOut', 'Agent deleted successfully!', 'success');
      loadAgents();
    } else {
      showNotification('createOut', 'Error deleting agent', 'error');
    }
  } catch (error) {
    showNotification('createOut', `Error: ${error.message}`, 'error');
  }
}

function editAgent(agentId) {
  const agent = agents.find(a => a.id === agentId);
  if (!agent) return;
  
  document.getElementById('name').value = agent.name;
  document.getElementById('sys').value = agent.system_prompt;
  document.getElementById('tools').value = agent.tools?.join(', ') || '';
  document.getElementById('model').value = agent.model || '';
  updateCharCount();
  
  // Scroll to top
  document.querySelector('section').scrollIntoView({ behavior: 'smooth' });
}

// File upload handling
function setupFileUpload() {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('files');
  const fileList = document.getElementById('fileList');
  
  // Click to browse
  dropZone.addEventListener('click', () => fileInput.click());
  
  // File input change
  fileInput.addEventListener('change', handleFileSelect);
  
  // Drag and drop
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });
  
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });
  
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files);
    handleFileSelect({ target: { files } });
  });
}

function handleFileSelect(event) {
  selectedFiles = Array.from(event.target.files);
  updateFileCount();
  displayFileList();
}

function displayFileList() {
  const fileList = document.getElementById('fileList');
  
  if (selectedFiles.length === 0) {
    fileList.innerHTML = '';
    return;
  }
  
  fileList.innerHTML = selectedFiles.map((file, index) => `
    <div class="flex items-center justify-between p-2 bg-gray-50 rounded">
      <div class="flex items-center space-x-2">
        <span class="text-sm">📄</span>
        <span class="text-sm text-gray-700">${file.name}</span>
        <span class="text-xs text-gray-500">(${(file.size / 1024).toFixed(1)} KB)</span>
      </div>
      <button onclick="removeFile(${index})" class="text-red-500 hover:text-red-700 text-sm">
        ✕
      </button>
    </div>
  `).join('');
}

function removeFile(index) {
  selectedFiles.splice(index, 1);
  updateFileCount();
  displayFileList();
}

async function uploadFiles() {
  const agentId = document.getElementById('agent').value;
  
  if (!agentId || selectedFiles.length === 0) {
    showNotification('upOut', 'Please select an agent and files', 'error');
    return;
  }
  
  try {
    const formData = new FormData();
    selectedFiles.forEach(file => formData.append('files', file));
    
    const response = await fetch(API + `/gems/${agentId}/files`, {
      method: "POST",
      body: formData
    });
    
    if (response.ok) {
      const result = await response.text();
      showNotification('upOut', 'Files uploaded successfully!', 'success');
      
      // Check KB status
      const statusResponse = await fetch(API + `/gems/${agentId}/kb/status`);
      const status = await statusResponse.text();
      document.getElementById('kbStat').textContent = status;
      
      // Clear files
      selectedFiles = [];
      document.getElementById('files').value = '';
      updateFileCount();
      displayFileList();
    } else {
      const error = await response.text();
      showNotification('upOut', `Error: ${error}`, 'error');
    }
  } catch (error) {
    showNotification('upOut', `Error: ${error.message}`, 'error');
  }
}

// Test agent functionality
async function testAgent() {
  const agentId = document.getElementById('testAgent').value;
  const message = document.getElementById('testMessage').value.trim();
  
  if (!agentId || !message) {
    showNotification('upOut', 'Please select an agent and enter a message', 'error');
    return;
  }
  
  try {
    const response = await fetch(API + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        gem_id: agentId,
        messages: [{ role: "user", content: message }],
        tools_mode: "auto"
      })
    });
    
    if (response.ok) {
      const result = await response.json();
      document.getElementById('testContent').textContent = result.content;
      document.getElementById('testResponse').classList.remove('hidden');
      
      if (result.used_tool) {
        document.getElementById('testContent').innerHTML += `
          <div class="mt-3 p-2 bg-blue-50 rounded text-sm">
            <strong>Used tool:</strong> ${result.used_tool}<br>
            <strong>Input:</strong> ${result.tool_input}
          </div>
        `;
      }
    } else {
      const error = await response.text();
      showNotification('upOut', `Error: ${error}`, 'error');
    }
  } catch (error) {
    showNotification('upOut', `Error: ${error.message}`, 'error');
  }
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
  // Character count for system prompt
  document.getElementById('sys').addEventListener('input', updateCharCount);
  
  // Agent selection for file upload
  document.getElementById('agent').addEventListener('change', updateFileCount);
  
  // Button event listeners
  document.getElementById('create').addEventListener('click', createAgent);
  document.getElementById('upload').addEventListener('click', uploadFiles);
  document.getElementById('sendTest').addEventListener('click', testAgent);
  document.getElementById('refreshAgents').addEventListener('click', loadAgents);
  
  // Test message input
  document.getElementById('testMessage').addEventListener('input', (e) => {
    document.getElementById('sendTest').disabled = !e.target.value.trim() || !document.getElementById('testAgent').value;
  });
  
  document.getElementById('testAgent').addEventListener('change', () => {
    document.getElementById('sendTest').disabled = !document.getElementById('testMessage').value.trim() || !document.getElementById('testAgent').value;
  });
  
  // Setup file upload
  setupFileUpload();
  
  // Initial load
  loadTemplates();
  loadAgents();
  updateCharCount();
});
</script>
</body></html>
    """
