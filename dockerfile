# app/Dockerfile

# Set base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Install git to clone the app
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone code that we put in the github repository
RUN git clone https://github.com/B-Robots-Belgium/RAG_Vito.git .

# Install dependencies
RUN pip3 install -r requirements.txt

# Expose the Docker container so that it listens on 8501 port
EXPOSE 8501

# This tells Docker how to test a container to check that it is still working
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Configures the docker to run as an executable.
ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]