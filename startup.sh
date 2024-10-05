#!/bin/bash

# Step 1: Execute the flask_conf.py file
python3 src/flask_conf.py &

# Step 2: Wait for 5 seconds
sleep 5

# Step 3: Open the default web browser to the localhost page
xdg-open http://127.0.0.1:5000