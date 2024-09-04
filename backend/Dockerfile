FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml /app/

RUN pip install poetry

RUN poetry config virtualenvs.create false && poetry install --no-root

# Install Azure CLI using the alternative method
RUN apt-get update && \
    apt-get install -y curl sudo && \
    curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash && \
    sudo apt-get update && sudo apt-get upgrade -y

COPY . /app

EXPOSE 80

CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
