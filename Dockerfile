# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and CSV files
COPY . .

# Expose the port (Render uses environment variable PORT, default 10000)
EXPOSE 10000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]