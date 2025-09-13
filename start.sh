#!/bin/bash
# Startup script for Hugging Face Spaces

# Set environment variables for Streamlit
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=7860
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Create streamlit config directory if it doesn't exist
mkdir -p ~/.streamlit

# Create streamlit config file
cat > ~/.streamlit/config.toml << EOL
[server]
headless = true
port = 7860
enableCORS = false
enableXsrfProtection = false
address = "0.0.0.0"

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
EOL

echo "🚀 Starting Atlan Customer Support Copilot..."
echo "📊 Running on port 7860"

# Start the Streamlit application
streamlit run app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
