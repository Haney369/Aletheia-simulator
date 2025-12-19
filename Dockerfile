FROM python:3.11-slim

# Set backend as the working directory
WORKDIR /app/backend

# Copy requirements first (better caching)
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ .

# Cloud Run port
ENV PORT=8080
EXPOSE 8080

# Start Flask app
CMD ["python", "-m", "app.app"]
