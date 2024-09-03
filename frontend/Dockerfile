FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml /app/

RUN pip install poetry

RUN poetry config virtualenvs.create false && poetry install --no-root

COPY . /app

EXPOSE 8501

CMD ["poetry", "run", "streamlit", "run", "app.py"]
