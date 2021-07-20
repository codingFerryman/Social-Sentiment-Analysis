import os
import re
from pathlib import Path

from transformers import PreTrainedModel


def get_project_path() -> Path:
    """The function for getting the root directory of the project"""
    try:
        import git
        return git.Repo(Path(), search_parent_directories=True).git.rev_parse("--show-toplevel")
    except NameError:
        return Path(__file__).parent.parent

def get_data_path() -> Path:
    return Path(get_project_path(), 'data')


def get_transformers_layers_num(model: PreTrainedModel) -> int:
    layer_numbers = []
    for name, _ in model.named_parameters():
        match = re.search('layer.(\d+)', name)
        if match:
            layer_numbers.append(int(match.group(1)))
    return max(layer_numbers)

def prepend_multiple_lines(file_name, list_of_lines):
    """Insert given list of strings as new lines at the beginning of a file"""
    # Create the file if it does not exist
    open(file_name, 'a').close()
    # define name of temporary dummy file
    dummy_file = str(file_name) + '.bak'
    # open given original file in read mode and dummy file in write mode
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        # Iterate over the given list of strings and write them to dummy file as lines
        for line in list_of_lines:
            write_obj.write(line + '\n')
        # Read lines from original file one by one and append them to the dummy file
        for line in read_obj:
            write_obj.write(line)
    os.remove(file_name)
    os.rename(dummy_file, file_name)
