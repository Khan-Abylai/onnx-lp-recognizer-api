#!/bin/bash

BACKUP_DIR="backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

tar -czf "$BACKUP_DIR/app_backup_$TIMESTAMP.tar.gz" \
    --exclude="*.pyc" \
    --exclude="__pycache__" \
    --exclude="logs/*" \
    --exclude="backups/*" \
    application/

tar -czf "$BACKUP_DIR/logs_backup_$TIMESTAMP.tar.gz" logs/

ls -t $BACKUP_DIR/app_backup_* | tail -n +6 | xargs -r rm
ls -t $BACKUP_DIR/logs_backup_* | tail -n +6 | xargs -r rm