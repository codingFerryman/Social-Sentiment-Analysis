#!/bin/bash
# Please connect to ETHZ internal network before using this script
declare ETH_USERNAME=$1
declare CIL_LOCALREPO=${2:-'~/cil-project'}

# Prepare folder and clone the repo
ssh $ETH_USERNAME@login.leonhard.ethz.ch '
REPOSRC=https://github.com/supernlogn/Computational-Intelligence-Lab.git
LOCALREPO='${CIL_LOCALREPO}'
LOCALREPO_VC_DIR=$LOCALREPO/.git
if [ ! -d $LOCALREPO_VC_DIR ]
then
    git clone $REPOSRC $LOCALREPO
else
    cd $LOCALREPO
    git pull $REPOSRC
fi
'

# ... or if you do not want to clone from GitHub but copy the project from local machine
#scp -r ../../src $USERNAME@login.leonhard.ethz.ch:cil-project

# Download the data if it not exists
ssh $ETH_USERNAME@login.leonhard.ethz.ch '
bash '${CIL_LOCALREPO}'/setup_dataset.sh
'

# Setup the environment
ssh $ETH_USERNAME@login.leonhard.ethz.ch '
bash '${CIL_LOCALREPO}'/setup_environment.sh
'
