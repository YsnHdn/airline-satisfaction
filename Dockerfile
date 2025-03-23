FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p models

# Download the dataset
RUN mkdir -p data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV API_URL=http://0.0.0.0:5000

# Expose ports for the API and Streamlit
EXPOSE 5000
EXPOSE 8501

# Train the model as part of the build process
RUN mkdir -p notebooks/trained_model && \
    python src/train_model.py

# Use a startup script to run both the API and Streamlit
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]