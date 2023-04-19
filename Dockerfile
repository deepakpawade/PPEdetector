
FROM python:3.9-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
# RUN apt-get update && apt-get install -y libgl1-mesa-glx && apt-get install libgl1
RUN python -m pip install -r requirements.txt
RUN python -m pip install opencv-python-headless

WORKDIR /app
COPY . /app


ENTRYPOINT ["python", "predict.py"]
# # Creates a non-root user with an explicit UID and adds permission to access the /app folder
# # For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# Set the entry point