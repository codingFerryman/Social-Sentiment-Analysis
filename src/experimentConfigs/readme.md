# Run experiments on Leonhard cluster
Please connect to the ETH network first.  

## Setup

### Setup SSH connection
It is recommended to install an SSH key pair to your Leonhard cluster such that your are not required to type your password every time. A step-by-step guide can be found here: https://scicomp.ethz.ch/wiki/Getting_started_with_clusters#SSH_keys  

### Setup the environment
1) Add Python3 to your cluster:  
    * Type `module avail python` to see the available Python modules
    * Load a Module by typing `module load python_cpu/3.7.1` (or any other available Python version)
    * Setup a virtual environment that contains all the packages needed

## Run the experiments
Once everything is setup you can run an experiment by executing: `./runExperimentOnLeonhard.sh ETH_USERNAME TEST_PATH`. 