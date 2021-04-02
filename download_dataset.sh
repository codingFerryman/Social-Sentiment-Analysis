#!/bin/bash
# Please connect to ETHZ internal network before using this script
mkdir -pv data
echo "*" > data/.gitignore
wget http://www.da.inf.ethz.ch/files/twitter-datasets.zip -O data/twitter-datasets.zip
cd data
unzip twitter-datasets.zip
rm twitter-datasets.zip
mv twitter-datasets/* ./
rm -d twitter-datasets
cd ..
