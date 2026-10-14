FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8001

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && groupadd --gid 10001 gateway \
    && useradd --uid 10001 --gid gateway --no-create-home gateway
COPY --chown=gateway:gateway ai_gateway ./ai_gateway
COPY --chown=gateway:gateway start.sh ./start.sh
RUN chmod +x start.sh && chown gateway:gateway /app

USER 10001:10001
EXPOSE 8001
CMD ["./start.sh"]
