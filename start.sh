#! /usr/bin/bash

source /apps/myproject/venv/bin/activate
uwsgi --ini /apps/myproject/uwsgi.ini
echo "Done"

