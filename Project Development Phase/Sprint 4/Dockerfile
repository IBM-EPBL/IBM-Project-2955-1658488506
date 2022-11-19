FROM python:3.9
RUN apt-get update
RUN mkdir /app
WORKDIR /app
COPY ./src /app/src
COPY ./requirements.txt /app
RUN pip install -r requirements.txt
EXPOSE 8000
ENTRYPOINT ["python"]
CMD ["main.py"]