# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container at /usr/src/app
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code
COPY main.py .
COPY config.py .
COPY .env .
COPY models/ ./models/
COPY metrics/ ./metrics/
COPY services/ ./services/
COPY storage/ ./storage/

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME AUTOSCALER_MODEL_SERVER

# Указываем команду для запуска приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]
