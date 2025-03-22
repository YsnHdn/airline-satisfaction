#!/bin/bash

# Start the Flask API in the background
python src/api.py &

# Start the Streamlit dashboard
streamlit run app/dashboard.py --server.port=8080 --server.address=0.0.0.0