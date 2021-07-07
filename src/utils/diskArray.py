import os
import numpy as np
import dill
from .loggers import getLogger
from dataclasses import dataclass

__all__ = ['DiskArray']

logger = getLogger("DiskArray", debug=True)

def getDiskArrayFileIndex():
    i = 0
    while True:
        i += 1
        yield i

DiskArrayFileIndexIterator = getDiskArrayFileIndex()


class DiskArray:
    def __init__(self):
        """ DiskArray is a list like object which stores its elements in disk memory rather than
        keeping them in RAM or cache. It uses the typical python files to do this. The file used is
        a binary file. The only methods supported are append and get item. Element deletion is not yet
        supported.
        - cusrorList: This is the list containing the cursor position in file where each element is stored
        - fileCursorEnd: This is the cursor position of the end of file
        - fileName: The name/path of the tempirary file where the DiskArray stores its elements. It is automotically assigned with an iterator value.
        - index: the index of the next element to be appended to the DiskArray.
        """
        self.cursorList: list = []
        self.fileCursorEnd: int = 0
        self.fileName: str = 'tempDiskArray{}.hdf5'.format(next(DiskArrayFileIndexIterator))
        self.index: int = 0

    def append(self, obj: any):
        """ This method appends an object to the file. It first transforms to a byte array and
        then stores its bytes to the file. The transformation is done by the dill library

        Args:
            obj (any): the object to store
        """
        logger.debug(f"append obj to {self.fileName}")
        with open(self.fileName, 'ab') as fw:
            self.cursorList.append(fw.tell())
            fw.write(dill.dumps(obj))
            self.fileCursorEnd = fw.tell()
            self.index += 1
    def __getitem__(self, key:int) -> any:
        """ This returns an object with an integer key provided

        Args:
            key (int): The index of the object to be retrieved from the file.

        Returns:
            any: The (key)-th object stored inside the DiskArray's file
        """
        logger.debug(f"Requesting key={key} from {self.fileName}")
        cursorPos = self.cursorList[key]
        el = None
        with open(self.fileName, 'rb') as fr:
            if key+1 < self.index:
                nextCursorPos = self.cursorList[key+1]
            else:
                nextCursorPos = self.fileCursorEnd
            fr.seek(cursorPos, 0)
            elBytes = fr.read(nextCursorPos - cursorPos)
            el = dill.loads(elBytes)
        return el
    def __del__(self):
        logger.debug(f"deleting {self.fileName}")
        if os.path.exists(self.fileName):
            os.remove(self.fileName)