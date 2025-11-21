# Neighbourhood AI Backend

FastAPI backend with OpenAI integration for neighbourhood insights.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key_here
   ```

3. Run locally:
   ```bash
   uvicorn main:app --reload
   ```

## API Endpoints

### POST /api/neighbourhood-story
Generates an AI-powered neighbourhood story based on address data.

**Request:**
```json
{
  "data": {
    "address": "Damrak 1, Amsterdam",
    "coords": [52.3738, 4.8907],
    "buildingInfo": {...},
    "cbsStats": {...}
  },
  "persona": "algemeen huishouden"
}
```

### POST /api/viewing-checklist
Generates a viewing preparation checklist.

**Request:**
```json
{
  "data": {...},
  "userPrefs": {
    "focus": "Ik let vooral op geluid, licht, staat van onderhoud..."
  }
}
```

## Deployment on Railway

1. Set the root directory to `/backend` (if using monorepo)
2. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Add environment variable: `OPENAI_API_KEY=your_key`
