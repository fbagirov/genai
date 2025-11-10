
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV SEARCH_PROVIDER=stub
CMD ["python","run_agent.py","--goal","Define AI in one sentence"]
