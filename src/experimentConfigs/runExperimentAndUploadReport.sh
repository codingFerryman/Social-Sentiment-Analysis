#!/bin/bash
declare TEST_PATH=$1

python experiment.py $TEST_PATH
sh uploadNewReport.sh