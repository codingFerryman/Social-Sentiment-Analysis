#!/bin/bash
if [[ ! -z $1 ]]; then
    REPORT_FILE=$1
else
    REPORT_FILE="../../docs/report.json"
fi
DIFF=`git diff $REPORT_FILE`
echo $DIFF # output diff
if [[ ! -z $DIFF ]]; then
    # upload the new report
    git add $REPORT_FILE
    git commit -m "doc: new report after running experiments"
    git push -u origin master
fi
