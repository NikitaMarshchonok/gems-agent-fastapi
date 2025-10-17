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

