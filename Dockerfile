FROM python:3.10-slim

# Install Java (required for PySpark)
RUN apt-get update && apt-get install -y openjdk-11-jdk && apt-get clean

# Set environment variable for Java
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . /app
WORKDIR /app

# Run Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=10000", "--server.enableCORS=false"]
