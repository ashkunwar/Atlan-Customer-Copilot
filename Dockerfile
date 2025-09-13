# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# Dockerfile for Atlan Customer Support Copilot on Hugging Face Spaces

FROM python:3.9-slim

# Create user with specific UID for Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY --chown=user ./requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application files
COPY --chown=user . /app

# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

# Create streamlit config directory
RUN mkdir -p /home/user/.streamlit

# Create streamlit config file to disable CORS and file watcher issues
RUN echo '[server]\n\
headless = true\n\
port = 7860\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
[browser]\n\
gatherUsageStats = false\n' > /home/user/.streamlit/config.toml

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "--browser.gatherUsageStats=false"]
