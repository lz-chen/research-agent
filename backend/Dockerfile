FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install poetry

RUN poetry install

EXPOSE 80

CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
