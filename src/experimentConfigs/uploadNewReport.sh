#!/bin/bash
gif diff report.json # output diff
# upload the new report
git add report.json
git commit -m "doc: new report after running experiments"
git push -u origin master
