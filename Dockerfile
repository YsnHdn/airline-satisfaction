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
RUN mkdir -p data && \
    python -c "import kaggle; kaggle.api.authenticate(); kaggle.api.dataset_download_files('teejmahal20/airline-passenger-satisfaction', path='./data', unzip=True)"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV API_URL=http://localhost:5000

# Expose ports for the API and Streamlit
EXPOSE 5000
EXPOSE 8080

# Train the model as part of the build process
RUN mkdir -p notebooks/trained_model && \
    python src/train_model.py

# Use a startup script to run both the API and Streamlit
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]