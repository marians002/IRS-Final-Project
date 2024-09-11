#!/bin/bash

# Step 1: Execute the flask conf.py file
python3 src/flask\ conf.py &

# Step 2: Wait for 10 seconds
sleep 10

# Step 3: Open the default web browser to the localhost page
xdg-open http://127.0.0.1:5000