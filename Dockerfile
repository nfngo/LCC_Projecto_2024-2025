
FROM python:3.11-alpine

WORKDIR /code

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

COPY assets .

CMD [ "python", "initial_setup.py" ]