# Use the official Python 3.10 image as the base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV USE_TORCH=1

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

RUN echo "import warnings; warnings.simplefilter('ignore')" > /usr/local/lib/python3.10/site-packages/sitecustomize.py

# Set a default command that can(should) be overridden
CMD ["python3.10", "/app/neurons/validator/main.py"]
