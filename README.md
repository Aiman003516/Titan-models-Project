# Titan ERP Generator Agent

Standalone agent for generating production-grade ERP modules using Titan fine-tuned models.

## Architecture

```
User Request → Titan Agent → RunPod (Titan-Backend) → Backend Code
                           → RunPod (Titan-UI)      → Frontend Code
                           → Groq (Llama 3.1 70B)   → Debug Fixes
                           → Validator               → Quality Checks
                           → ZIP Package             → Downloadable Output
```

## Stack

- **Tier 1 Models**: Titan-Backend + Titan-UI (Qwen3-32B, DoRA fine-tuned)
- **Tier 2 Debugger**: Groq (Llama 3.1 70B, free, 2-5s responses)
- **Infrastructure**: RunPod Serverless (vLLM), Vultr VPS
- **Backend**: FastAPI + WebSocket
- **Frontend**: Nuxt 3 + PrimeVue 4 + Tailwind CSS (generated code)

## Quick Start

```bash
# Clone
git clone https://github.com/Aiman003516/Titan-models-Project.git
cd Titan-models-Project

# Setup
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
nano .env  # Add your API keys

# Run
python -m uvicorn titan.main:app --host 0.0.0.0 --port 8000

# Open http://localhost:8000
```

## VPS Deployment

```bash
# On the VPS as root:
bash setup_vps.sh
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/health` | Health check |
| GET | `/api/modules` | List available modules |
| POST | `/api/build` | Start ERP build |
| GET | `/api/build/{id}` | Poll build status |
| GET | `/api/build/{id}/download` | Download ZIP |
| WS | `/ws/build?modules=crm,hr` | Real-time build events |

## Modules

45 ERP modules across 5 architecture patterns:
- **Service Layer** (16): CRM, HR, Fleet, Payment, etc.
- **DDD** (20): Account, Sale, Purchase, Stock, etc.
- **CQRS** (4): Helpdesk, Bank Reconciliation, etc.
- **Hexagonal** (3): Warehouse, Documents, Website
- **Event Sourced** (1): Manufacturing (MRP)
