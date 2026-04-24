# AgenticAI-Trip-Planner-Microsoft-demo

## Test Agent 1 with your own data

Agent 1 (Place Recommender) now includes a manual runner:

- Script: `agent1-place-recommender/manual_test_agent1.py`
- Sample payload: `agent1-place-recommender/sample-input.json`

### Steps

1. Edit `agent1-place-recommender/sample-input.json` with your own trip data.
2. Make sure Agent 1 is running (for local Docker compose this is typically on `http://localhost:8001`).
3. Run the manual script and point it at your payload.

### What you get

- Full JSON response from Agent 1.
- Printed `place_candidates` count for quick validation.

### Notes

- If `AGENT1_URL` in `.env` is set to `http://agent1:8001` and you are running from host shell,
  use `http://localhost:8001` instead.
- If Google Geocoding is not enabled for your API key/project, Agent 1 may fail early.
