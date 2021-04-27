# Computational Intelligence Lab Project 2
This is the project 2 of CIL course in ETHZ.

## Developing
To work for this project create a virtual environment in a folder outside the repo-path or use .gitignore to ignore any files of the virtual envronment folder.

### Code Formatting

For code formatting please do these steps:
1. install yapf: `pip install yapf`
2. ~~Download the yapf configuration file at and add it to `<your-repo-path>/.style.yapf`~~
3. Download the pre-commit.sh file from this repo and add it to `<your-repo-path>/.git/hooks/pre-commit`
4. Make sure pre-commit is marked as executable

When commiting always enable virtual environment so that the OS can find yapf.
For more information about yapf: See here: https://github.com/google/yapf

Do any contributions to yapf but always anounce your decisions to the team.
