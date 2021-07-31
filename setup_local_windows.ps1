# This script is for setting up the environment on Windows
# You should have python3.7+ installed on your computer

# If you meet any problems about missing DLL files (except CUDA libraries), please try to install
# ... VC++ Redistributable for VS: https://aka.ms/vs/16/release/vc_redist.x64.exe
# ... or anaconda on your computer


$CIL_LOCALREPO=$ExecutionContext.SessionState.Path.CurrentFileSystemLocation

if ((Get-ChildItem $CIL_LOCALREPO\data\*.txt -File | Measure-Object).Count -eq 5){
    Write-Host "Data files already downloaded."
} else {
    Write-Host "Downloading data files ..."
    Remove-Item $CIL_LOCALREPO\data\*.txt
    Remove-Item $CIL_LOCALREPO\data\*.zip
    Invoke-RestMethod http://www.da.inf.ethz.ch/files/twitter-datasets.zip -OutFile $CIL_LOCALREPO\data\twitter-datasets.zip
    Expand-Archive -LiteralPath $CIL_LOCALREPO\data\twitter-datasets.zip -DestinationPath $CIL_LOCALREPO\data
    Move-Item $CIL_LOCALREPO\data\twitter-datasets\*.txt $CIL_LOCALREPO\data
    Remove-Item $CIL_LOCALREPO\data\twitter-datasets -Recurse
    Remove-Item $CIL_LOCALREPO\data\twitter-datasets.zip
    Write-Host "Downloaded!"
}

if (Test-Path -Path .git) {
    Write-Host ".git directory already exists."
} else {
    git init
}

if (Test-Path -Path $CIL_LOCALREPO\venv) {
    Write-Host "Virtual environment already exists."
} else {
    python -m venv $CIL_LOCALREPO\venv
}
$ACTIVATE_VENV=Join-Path $CIL_LOCALREPO \venv\Scripts\Activate.ps1
. $ACTIVATE_VENV

python -m pip install --upgrade pip setuptools wheel

# CUDA
#pip install -r "${SCRIPT_DIR}"/requirements.txt
# CPU
pip install -r "${SCRIPT_DIR}"/requirements_cpu.txt

#spacy download en_core_web_sm
#spacy download en_core_web_trf
#python "${SCRIPT_DIR}"/setup.py install
