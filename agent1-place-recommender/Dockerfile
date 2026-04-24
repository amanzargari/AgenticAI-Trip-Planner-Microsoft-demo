FROM python:3.11-slim

WORKDIR /app

COPY shared/ ./shared/
COPY agent1-place-recommender/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY agent1-place-recommender/ ./

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
