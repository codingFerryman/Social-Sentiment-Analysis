#!/bin/bash
# Please connect to ETHZ internal network before using this script
declare USERNAME=$1
declare TEST_PATH=$2
HARDWARE_REQUIREMENTS='-n 8 -R "rusage[mem=12000,scratch=10000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]"'
# Prepare folder
# ssh $USERNAME@login.leonhard.ethz.ch 'rm -r ~/cil-project; mkdir ~/cil-project'

# Copy the project
# scp -r ../../src $USERNAME@login.leonhard.ethz.ch:cil-project

# Download the data and run the experiment
COMMANDS='cd Computational-Intelligence-Lab
git pull
bash setup_dataset.sh
cd src/experimentConfigs
chmod +x runExperimentAndUploadReportCluster.sh
bsub -W 23:30'
COMMANDS+=' '
COMMANDS+=$HARDWARE_REQUIREMENTS
COMMANDS+=' '
COMMANDS+='./runExperimentAndUploadReportCluster.sh'
COMMANDS+=' '
COMMANDS+=$TEST_PATH
# echo for inspection
# echo $COMMANDS
ssh $USERNAME@login.leonhard.ethz.ch "$COMMANDS"