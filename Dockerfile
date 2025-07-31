FROM python:3.11-slim

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    libglib2.0-0 \
    libnss3 \
    libatk-bridge2.0-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libxshmfence1 \
    libgtk-3-0 \
    wget \
    && apt-get clean

# Install playwright dependencies and browser
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install --with-deps

# Copy the app
COPY ./app /

# Expose port
EXPOSE 8080

# Run FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
