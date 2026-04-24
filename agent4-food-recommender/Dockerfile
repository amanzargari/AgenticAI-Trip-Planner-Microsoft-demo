FROM python:3.11-slim

WORKDIR /app

# shared utilities (needed by all agents)
COPY shared/ ./shared/

# agent-specific files
COPY agent4-food-recommender/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY agent4-food-recommender/ ./

EXPOSE 8004
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
