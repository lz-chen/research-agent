FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install poetry

RUN poetry install

EXPOSE 80

CMD ["poetry", "run", "streamlit", "run", "app.py"]
