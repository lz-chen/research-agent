FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml poetry.lock /app/

RUN pip install poetry

RUN poetry install --no-root

COPY . /app

RUN pip install poetry

RUN poetry install

EXPOSE 80

CMD ["poetry", "run", "streamlit", "run", "app.py"]
