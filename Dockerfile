# Set base image (host OS)
FROM python:3.9-slim

# By default, listen on port 5000
EXPOSE 5001/tcp

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN apt-get update \
  && apt-get -y install tesseract-ocr \
  && apt-get -y install ffmpeg libsm6 libxext6
RUN pip install -r requirements.txt


# Copy the content of the local src directory to the working directory
ENV PYTHONPATH "${PYTHONPATH}:/code/"
COPY *.py .
COPY .env .



# Specify the command to run on container start
CMD [ "python", "./app.py" ]