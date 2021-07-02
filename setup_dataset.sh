#!/bin/bash
# Please connect to ETHZ internal network before using this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck disable=SC2012
DATA_FILE_COUNT=$(ls "${SCRIPT_DIR}"/data/*.txt | wc -l)

if [ "$DATA_FILE_COUNT" != "5" ]; then
  echo "Downloading data files ..."
  rm "${SCRIPT_DIR}"/data/*.txt
  wget http://www.da.inf.ethz.ch/files/twitter-datasets.zip -O "${SCRIPT_DIR}"/data/twitter-datasets.zip
  unzip "${SCRIPT_DIR}"/data/twitter-datasets.zip -d "${SCRIPT_DIR}"/data
  rm "${SCRIPT_DIR}"/data/twitter-datasets.zip
  echo "Downloaded!"
else
  echo "Data files already downloaded."
fi
