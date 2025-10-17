# Gems Agent (FastAPI)

A lightweight "Gems-like" agent server (profiles, tool-use) built on FastAPI with Ollama as the default LLM backend. Comes with ready-made profiles (Travel, Code Helper, English Tutor) and tools (web_search, calculator).


## 📸 Screenshots


Project structure

![Project structure](pics/1.png)


Ollama version & models

![Ollama version & models](pics/2.png)


.env (no secrets

![.env (no secrets)](pics/3.png)


Healthcheck OK

![Healthcheck OK](pics/5.png)


Chat result (itinerary)

![Chat result (itinerary)](pics/4.png)

Tip: If you prefer an English chat send the prompt: “Plan a 2-day itinerary in Paris. Include logistics and approximate prices in EUR. Answer in English only.”


##  Quick Start
Requirements

Python 3.11+

macOS/Linux

Ollama running locally


1) Setup
   
   python3 -m venv .venv

   source .venv/bin/activate

   pip install -r requirements.txt

2) Ollama

   install & start (Homebrew on macOS):

   brew install ollama

   brew services start ollama

   ollama pull llama3.1:8b

3) Environment

   cp .env.example .env

   Open .env and set:

   LLM_BACKEND=ollama
   
   OLLAMA_BASE_URL=http://127.0.0.1:11434
   
   OLLAMA_MODEL=llama3.1:8b


4) Run the server

   uvicorn app.main:app --reload --port 8000

   You should see: Uvicorn running on http://127.0.0.1:8000 and Application startup complete.



## Smoke Test (copy-paste)

In a new terminal (while the server is running):
      
      health
      curl -s http://127.0.0.1:8000/health | jq


      list profiles ("gems")
      curl -s http://127.0.0.1:8000/gems | jq


      pick Travel id
      TRAVEL_ID=$(curl -s http://127.0.0.1:8000/gems | jq -r '.[] | select(.name=="Travel") | .id')


      echo "TRAVEL_ID=$TRAVEL_ID"


      chat (English)
      cat > req_en.json <<'JSON'
      {
      "gem_id": "REPLACE_ME",
      "messages": [
      { "role": "user", "content": "Plan a 2-day itinerary in Paris. Include logistics and approximate prices in EUR. Answer in English only." }
      ],
      "tools_mode": "auto"
      }
      JSON


      sed -i '' "s/REPLACE_ME/$TRAVEL_ID/" req_en.json 2>/dev/null || sed -i "s/REPLACE_ME/$TRAVEL_ID/" req_en.json


      pretty print only the text content (use jq)
      curl -s -X POST http://127.0.0.1:8000/chat \
      -H 'Content-Type: application/json' \
      --data-binary @req_en.json | jq -r '.content'

If you don’t have jq, use Python to print Unicode nicely:

      curl -s -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' --data-binary @req_en.json \
      | python - <<'PY'
      import sys, json
      print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))
      PY

API Overview

   GET /health — service status

   GET /gems — list profiles

   GET /gems/{id} — get a profile

   PUT /gems/{id} — update profile fields (model, temperature, system_prompt)

   POST /chat — chat with a profile



POST /chat body

      {
      "gem_id": "<uuid>",
      "messages": [{"role":"user","content":"..."}],
      "tools_mode": "auto" | "off"
      }

Response (simplified):

      {
      "content": "... final text ...",
      "used_tool": null | "web_search" | "calculator",
      "tool_input": null | "..."
      }


Open interactive docs:    http://127.0.0.1:8000/docs



Notes & Tips

   Model quality matters. llama3.1:8b is a good local default; smaller models may hallucinate.

   To force English responses by default, you can update the Travel profile:

      TRAVEL_ID=$(curl -s http://127.0.0.1:8000/gems | jq -r '.[] | select(.name=="Travel") | .id')
      
      awk 'BEGIN{printf "{"}{printf "\"system_prompt\":\"You are a world-class travel planner. Always respond in English. Be concise, structured, and pragmatic.\""; printf "}"}' | \
      
      curl -s -X PUT "http://127.0.0.1:8000/gems/$TRAVEL_ID" -H 'Content-Type: application/json' --data-binary @- | jq
