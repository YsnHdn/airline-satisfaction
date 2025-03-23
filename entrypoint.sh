#!/bin/bash

# Start the Flask API in the background
python src/api.py &

# Wait for API to start
sleep 5

# Start the Streamlit dashboard with explicit parameters
# Ces options assurent que Streamlit écoute sur toutes les interfaces
streamlit run app/dashboard.py --server.port=8501 --server.address=0.0.0.0 --server.enableCORS=true --server.enableXsrfProtection=false