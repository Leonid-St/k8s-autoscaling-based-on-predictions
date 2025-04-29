FROM python:3.7

COPY requirements.txt /app/
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY wsgi.py .
COPY start.sh .
RUN chmod +x ./start.sh

ENTRYPOINT ["./start.sh"]
