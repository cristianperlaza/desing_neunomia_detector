FROM python:latest

RUN apt-get update -y && \
    apt-get install python3-opencv -y 

COPY requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt
WORKDIR /app

RUN curl "https://www.dropbox.com/s/yx6n606i7cfcvoz/WilhemNet_86.h5?dl=1" -L -o WilhemNet_86.h5
COPY . ./

ENTRYPOINT ["python"]
CMD ["detector_neumonia.py"]