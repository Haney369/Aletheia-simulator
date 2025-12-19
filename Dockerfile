# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy backend requirements
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Cloud Run expects PORT
ENV PORT=8080
ENV PYTHONPATH=/app

# Expose the port
EXPOSE 8080

# Start Flask app
CMD ["python", "app.py"]

