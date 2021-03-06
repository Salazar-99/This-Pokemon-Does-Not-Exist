#!/bin/bash

#Load .env file containing AWS credentials
set -a
source .env
set +a

#Make directory for AWS credentials in default location if it does not already exist
if [[ ! -d ~/.aws/ ]]
then
    mkdir ~/.aws/
fi

#Switch to AWS directory and create credentials file
cd ~/.aws/
touch credentials

#Write AWS credentials to appropriate file
filepath=~/.aws/credentials
if [[ -f "$filepath" ]]
then 
    > "$filepath"
    echo "[default]" >> "$filepath"
    echo "aws_access_key_id = $ACCESS_KEY_ID" >> "$filepath"
    echo "aws_secret_access_key = $SECRET_ACCESS_KEY" >> "$filepath"
fi
