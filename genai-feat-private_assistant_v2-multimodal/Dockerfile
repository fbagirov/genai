FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV CONFIG_PATH=/app/configs/config.yaml \
    API_URL=http://localhost:8000/api/chat

EXPOSE 8000 7860
CMD ["bash", "-lc", "python -m app.main & python -m ui.chat_ui"]
