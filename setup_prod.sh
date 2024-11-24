#!/bin/bash

mkdir -p prod
mkdir -p prod/src
mkdir -p prod/models
mkdir -p prod/logs

touch prod/src/__init__.py

cp -r application/src/* prod/src/
cp -r application/models/* prod/models/

cp docker-compose.yml prod/
cp application/Dockerfile prod/
cp .env prod/

touch prod/logs/app.log