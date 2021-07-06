#!/bin/bash
declare TEST_PATH=$1

python experiment.py test_path=$TEST_PATH report_path='../../docs/report.json'
sh uploadNewReport.sh