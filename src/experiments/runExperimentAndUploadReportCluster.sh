#!/bin/bash
# This script is executed inside the cluster. Its purpose is to load the modules and environment needed
# by the scripts to run, execute the training and then upload the report of the training.
# This should be executed inside the cluster using the bsub command.
# To submit this while inside the cluster you can type:
#
# bsub -W 23:30 -n 8 -R "rusage[mem=9000,scratch=10000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000] \
# ./runExperimentAndUploadReportCluster.sh <path-to-configuration-inside-Leonhard>
#

declare TEST_PATH=$1
module load gcc/6.3.0 python_gpu/3.8.5 hdf5/1.10.1 eth_proxy
source cil-venv/bin/activate
python experiment.py test_path=$TEST_PATH report_path='../../docs/report.json'
sh uploadNewReport.sh