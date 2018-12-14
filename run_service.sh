#!/bin/bash
source venv/bin/activate
exec gunicorn -b 0.0.0.0:5000 --access-logfile - --error-logfile - keras_api:app
