# Support Ticket Intelligence

An AI-powered customer support system that automatically classifies, summarizes, and routes customer support tickets to reduce manual work and response time.

### Features

- Auto-classification using BERT + LoRA fine-tuning
- AI-generated summaries with Mistral
- Suggested replies for faster response times
- Snowflake integration for ticket storage

### How to run

#### Clone the Repository
```bash
git clone https://github.com/nethra4321/support-ticket-intelligence.git
cd support-ticket-intelligence
```


### Backend Setup

#### Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

#### Install dependencies
```bash
pip install -r requirements.txt
```

#### Configure environment

Create `.env` in project root:
```env
# Snowflake Configuration
SNOWFLAKE_ACCOUNT=account
SNOWFLAKE_USER=username
SNOWFLAKE_PASSWORD=password
SNOWFLAKE_ROLE=role
SNOWFLAKE_WAREHOUSE=warehouse
SNOWFLAKE_DATABASE=STI
SNOWFLAKE_SCHEMA=PUBLIC

# LLM Configuration
OLLAMA_HOST=http://localhost:11434
```

#### Start the backend
```bash
uvicorn backend.main:app --reload --port 8000
```

Backend runs at `http://localhost:8000`


### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:3000`

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/tickets` | List all tickets |
| GET | `/tickets/{id}` | Get ticket details |
| POST | `/tickets/{id}/generate` | Generate summary & reply |
| POST | `/tickets/{id}/classify` | Classify ticket |

### Requirements

- Python 3.9+
- Node.js 18+
- Snowflake account with credentials
- Ollama  for LLM inference

### Models

- Classification: BERT + LoRA fine-tuned model
- Summarization & response: Mistral
 
