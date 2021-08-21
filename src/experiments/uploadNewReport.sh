#!/bin/bash
# This scripts uploads the report.json to github by commiting it and pushing to the branch that the
# local github project is currently on. It accepts one argument the path of the report file.
# The argument by default is ../../docs/report.json
#

if [[ ! -z $1 ]]; then
    REPORT_FILE=$1
else
    REPORT_FILE="../../docs/report.json"
fi
# if [[ ! "$REPORT_FILE" == *json ]] then
#     echo "Please specify a json file for final report"
#     exit 0
# fi
DIFF=`git diff $REPORT_FILE`
echo $DIFF # output diff
if [[ ! -z $DIFF ]]; then
    # upload the new report
    git add $REPORT_FILE
    git commit -m "doc: new report after running experiments"
    git push -u origin master
fi
