import os
import numpy as np
import dill as pickle
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
    cursorList: list = []
    fileCursorEnd: int = 0
    fileName: str = 'tempDiskArray{}.hdf5'.format(next(DiskArrayFileIndexIterator))
    index: int = 0

    def append(self, obj: any):
        logger.debug(f"append obj to {self.fileName}")
        with open(self.fileName, 'ab') as fw:
            self.cursorList.append(fw.tell())
            fw.write(pickle.dumps(obj))
            self.fileCursorEnd = fw.tell()
            self.index += 1
    def __getitem__(self, key:int) -> any:
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
            el = pickle.loads(elBytes)
        return el
    def __del__(self):
        logger.debug(f"deleting {self.fileName}")
        if os.path.exists(self.fileName):
            os.remove(self.fileName)