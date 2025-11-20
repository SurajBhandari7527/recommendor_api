# Use python slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# --- NEW: Install system dependencies for Scikit-Learn ---
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# ---------------------------------------------------------

# Copy requirements first
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port
EXPOSE 10000

# Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]