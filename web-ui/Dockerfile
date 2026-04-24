FROM python:3.11-slim

WORKDIR /app

COPY shared/ ./shared/
COPY web-ui/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY web-ui/ ./

EXPOSE 3000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"]
