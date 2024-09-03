FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml /app/

RUN pip install poetry

RUN poetry config virtualenvs.create false && poetry install --no-root

COPY . /app

EXPOSE 80

CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
