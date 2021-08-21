#!/bin/bash
# This script loads the environment needed to execute the experiment.py and 
# then executes the training of the model. After the training ends, 
# the results are stored inside docs/report.json and uploaded to github.
# This scripts also assumes that a cil-venv environment exists in the base
# of the repo and tries to load it.
#


declare TEST_PATH=$1
source cil-venv/bin/activate
python experiment.py test_path=$TEST_PATH report_path='../../docs/report.json'
sh uploadNewReport.sh