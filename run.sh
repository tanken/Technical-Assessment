#!/usr/bin/env bash

pip install -r requirements.txt

echo "Please select your algorithm: "
echo "1 - Linear Regression"
echo "2 - Lasso Regression"
echo "3 - Ridge Regression"
echo "4 - END program"

echo "You choice: "
read user_algo

python mlp/program.py $user_algo
