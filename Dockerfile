FROM python:3.6-stretch

ENV FLASK_APP keras_api.py

RUN adduser --disabled-password -q keraspred
USER keraspred
WORKDIR /home/keraspred

COPY requirements.txt ./
RUN python -m venv venv
RUN venv/bin/pip install -r requirements.txt

COPY keras_api.py run_service.sh ./
RUN pwd
RUN ls

EXPOSE 5000
ENTRYPOINT ["./run_service.sh"]
