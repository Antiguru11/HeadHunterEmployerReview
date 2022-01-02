FROM --platform=amd64 continuumio/miniconda3:latest

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl wget

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt --user