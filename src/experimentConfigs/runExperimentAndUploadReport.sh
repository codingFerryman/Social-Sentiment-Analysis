#!/bin/bash
declare TEST_PATH=$1
module load gcc/6.3.0 python_gpu/3.8.5 hdf5/1.10.1 eth_proxy
source cil-venv/bin/activate
python experiment.py test_path=$TEST_PATH report_path='../../docs/report.json'
sh uploadNewReport.sh