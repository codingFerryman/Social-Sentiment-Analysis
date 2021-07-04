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

if (Test-Path -Path $CIL_LOCALREPO\venv) {
    Write-Host "Virtual environment already exists."
} else {
    python -m venv $CIL_LOCALREPO\venv
}
$ACTIVATE_VENV=Join-Path $CIL_LOCALREPO \venv\Scripts\Activate.ps1
. $ACTIVATE_VENV

python -m pip install --upgrade pip setuptools wheel

# If you have CUDA installed on your computer
#pip install torch==1.8.1+cu102 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
#pip install --upgrade 'spacy[cuda102]'

# If you DON'T have CUDA installed on your computer
pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install --upgrade 'spacy' # If you DON'T have CUDA installed on your computer

pip install 'torchtext<0.10'
spacy download en_core_web_sm
pip install -r $CIL_LOCALREPO\requirements.txt
