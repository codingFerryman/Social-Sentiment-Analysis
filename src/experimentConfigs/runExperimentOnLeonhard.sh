#!/bin/bash
# Please connect to ETHZ internal network before using this script
declare USERNAME=$1
declare TEST_PATH=$2

# Prepare folder
ssh $USERNAME@login.leonhard.ethz.ch 'rm -r ~/cil-project; mkdir ~/cil-project'

# Copy the project
scp -r ../../src $USERNAME@login.leonhard.ethz.ch:cil-project

# Download the data and run the experiment
ssh $USERNAME@login.leonhard.ethz.ch 'cd cil-project
wget http://www.da.inf.ethz.ch/files/twitter-datasets.zip
unzip twitter-datasets.zip
rm twitter-datasets.zip
mv twitter-datasets data
cd src/experimentConfigs
bsub < runExperimentAndUploadReport.sh $TEST_PATH'