FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV USE_TORCH=1
ENV AXON_PORT=8101

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

RUN echo "import warnings; warnings.simplefilter('ignore')" > /usr/local/lib/python3.10/site-packages/sitecustomize.py

# Set a default command that can(should) be overridden
CMD ["python3.10", "/app/neurons/miners/StableMiner/main.py"]
