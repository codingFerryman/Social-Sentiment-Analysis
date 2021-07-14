import os
import numpy as np
import dill
import typing
from collections.abc import Sequence
from .loggers import getLogger
from dataclasses import dataclass

__all__ = ['DiskArray']

logger = getLogger("DiskArray", debug=False)

def getDiskArrayFileIndex() -> str:
    """return index for a new DiskArrayFile. Index is increased by one each time a new file is created"""
    i = 0
    while True:
        i += 1
        fileName = f'tempDiskArray{i}.hdf5'
        if os.path.exists(fileName):
            continue 
        yield fileName

DiskArrayFileIndexIterator = getDiskArrayFileIndex()


class DiskArray(Sequence):
    BYTEORDER:str = 'big'
    INTSIZE:int = 32
    BIG_INTSIZE:int= 64
    INTSIZE_BYTES:int = 32//4
    BIG_INTSIZE_BYTES:int= 64//4

    def __init__(self, fileName:str=None):
        """ DiskArray is a list like object which stores its elements in disk memory rather than
        keeping them in RAM or cache. It uses the typical python files to do this. The file used is
        a binary file. The only methods supported are append and get item. Element deletion is not yet
        supported.
        - cusrorList: This is the list containing the cursor position in file where each element starts to be stored. First element is zero(0).
        - fileCursorEnd: This is the cursor position of the end of file
        - fileName: The name/path of the tempirary file where the DiskArray stores its elements. It is automotically assigned with an iterator value.
        - numObjs: the number of objects stored inside the DiskArray instance.
        """
        self.cursorList: list = []
        self.fileCursorEnd: int = 0
        self.fileName: str = next(DiskArrayFileIndexIterator) if fileName == None else fileName
        
        self.numObjs: int = 0

    def appendBytes(self, _bytes:bytes):
        """This method appends bytes of an object to the file. The transformation to bytes is done by the dill library.
        For keeping and indexing as an array all the necessary indexes and pointers are updated.

        Args:
            _bytes (bytes): bytes of object to append to diskArray
        """
        logger.debug(f"append bytes to {self.fileName}")
        with open(self.fileName, 'ab') as fw:
            self.cursorList.append(fw.tell())
            fw.write(_bytes)
            self.fileCursorEnd = fw.tell()
            self.numObjs += 1

    def append(self, obj: any):
        """ This method appends an object to the file. It first transforms to a byte array and
        then stores its bytes to the file. The transformation is done by the dill library

        Args:
            obj (any): the object to store
        """
        logger.debug(f"append obj to {self.fileName}")
        self.appendBytes(dill.dumps(obj))

    def __len__(self):
        return self.numObjs

    def __sizeof__(self):
        return self.fileCursorEnd
    
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
            if key+1 < self.numObjs:
                nextCursorPos = self.cursorList[key+1]
            else:
                nextCursorPos = self.fileCursorEnd
            fr.seek(cursorPos, 0)
            elBytes = fr.read(nextCursorPos - cursorPos)
            el = dill.loads(elBytes)
        return el
    
    def iterateBytes(self) -> bytes:
        """ Iterates the bytes of the elements of this diskArray.
        This function is a generator.

        Returns:
            typing.Iterator: iterator to the iteration of the cursorList 

        Yields:
            bytes: bytes from iterating the bytes of the elements of this diskArray.
        """
        i:int = 0
        cursorPos = 0
        with open(self.fileName, 'rb') as fr:
            fr.seek(self.cursorList[0], 0)
            while i+1 < self.numObjs: # allow content to be added dynamically while iterating
                cursorPos = self.cursorList[i]
                nextCursorPos = self.cursorList[i+1]
                elBytes = fr.read(nextCursorPos - cursorPos)
                i += 1
                yield elBytes
            # write last element to file
            if self.fileCursorEnd > 0:
                fr.seek(cursorPos, 0)
                elBytes = fr.read(self.fileCursorEnd - cursorPos)
                yield elBytes

    def __iter__(self):
        """ Iterates the elements of this diskArray.
        This function is a generator.
        Yields:
            any: Object returned by iterating the elements of this diskArray
        """
        for _bytes in self.iterateBytes():
            yield dill.loads(_bytes)

    def iterateBytesReversed(self) -> bytes:
        """ Iterates the bytes of the elements of this diskArray in reversed order.
        This function is a generator.

        Returns:
            typing.Iterator: iterator to the iteration of the cursorList in reversed order 

        Yields:
            bytes: bytes from iterating the bytes of the elements of this diskArray in reversed order.
        """
        i:int = self.numObjs - 1
        if self.fileCursorEnd > 0:
            with open(self.fileName, 'rb') as fr:
                cursorPos = self.cursorList[i]
                nextCursorPos = self.fileCursorEnd
                fr.seek(cursorPos, 0)
                elBytes = fr.read(nextCursorPos - cursorPos)
                yield elBytes
                i = i -1
                while i >= 0: # allow content to be added dynamically while iterating
                    cursorPos = self.cursorList[i]
                    nextCursorPos = self.cursorList[i+1]
                    elBytes = fr.read(nextCursorPos - cursorPos)
                    i -= 1
                    yield elBytes

    def __reversed__(self):
        """ Iterates the elements of this diskArray in reversed order
        This function is a generator.
        Yields:
            any: object stored inside the diskArray
        """
        for _bytes in self.iterateBytesReversed():
            yield dill.loads(_bytes)
        
    def containsBytes(self, _bytes:bytes)->bool:
        """If this diskArray contains an object transformable to these bytes.
        
        Args:
            _bytes (bytes): The bytes to be searched for

        Returns:
            bool: Whether this diskArray contains an object transformable to these bytes
        """
        for elbytes in self.iterateBytes():
            if elbytes == _bytes:
                return True
        return False

    
    def __contains__(self, value: any) -> bool:
        """ If this diskArray contains an object with a value provided

        Args:
            value (any): value of object to search for

        Returns:
            bool: Whether this diskArray contains such an object
        """
        vbytes = dill.dumps(value)
        return self.containsBytes(vbytes)
    
    def indexBytes(self, _bytes:bytes) -> int:
        """Find first index of an object transformable to _bytes inside diskArray

        Args:
            _bytes (bytes):  The bytes to be searched for

        Raises:
            ValueError: In case no such object exists

        Returns:
            int: The first index of an object transformable to _bytes inside diskArray
        """
        i:int = 0
        for elbytes in self.iterateBytes():
            if elbytes == _bytes:
                return i
            i += 1
        raise ValueError(f"bytes: {_bytes} do not exist")

    def index(self, x:any) -> int:
        """Find first index of an object equal to a value provided

        Args:
            x (any): value of object to search for

        Raises:
            Value: In case no such object exists

        Returns:
            int: The first index of an object equal to the value provided
        """
        try:
            i = self.indexBytes(dill.dumps(x))
            return i
        except ValueError:
            raise Value(f"object {x} does not exist in DiskArray")
        
    def save(self, newFilePath:str):
        """This saves the current file used by this diskArray instance to a file newFilePath.
        Attention: This file is not the same file used by diskArray!!!!

        Args:
            newFilePath (str): file path to the new file
        """
        logger.info(f"Saving DiskArray from {self.fileName} to {newFilePath}")
        try:
            with open(newFilePath, 'wb') as fw:
                fw.write(self.numObjs.to_bytes(DiskArray.INTSIZE, byteorder=DiskArray.BYTEORDER, signed=False))
                fw.write(self.fileCursorEnd.to_bytes(DiskArray.BIG_INTSIZE, byteorder=DiskArray.BYTEORDER, signed=False))
                for s in self.cursorList:
                    fw.write(s.to_bytes(DiskArray.BIG_INTSIZE, byteorder=DiskArray.BYTEORDER, signed=False))
                # cursorIndex = fw.tell()
                for _bytes in self.iterateBytes():
                    fw.write(_bytes)
        except OSError as e:
            raise f"{e.strerror} No such file found for saving DiskArray"

    def load(filePath:str) -> 'DiskArray':
        """Load a diskArray from a file saved via the diskArray.save function.
        Attention: This file is not the same file used by diskArray!!!!

        Args:
            filePath (str): file path to load the diskArray from

        Returns:
            DiskArray: The diskArray recovered from the file. This diskArray will use another file to store its data.
        """
        logger.info(f"Loading DiskArray from {filePath}")
        newD = DiskArray()
        try:
            with open(filePath, 'rb') as fr:
                index = int.from_bytes(fr.read(DiskArray.INTSIZE_BYTES), byteorder=DiskArray.BYTEORDER)
                fileCursorEnd = int.from_bytes(fr.read(DiskArray.BIG_INTSIZE_BYTES), byteorder=DiskArray.BYTEORDER)
                cursorList = [int.from_bytes(fr.read(DiskArray.BIG_INTSIZE_BYTES), byteorder=DiskArray.BYTEORDER) for i in range(index)]
                currentCursor = fr.tell()
                tempCursorList = [c+currentCursor for c in cursorList] + [fileCursorEnd + currentCursor]
                for i in range(len(tempCursorList)-1):
                    newD.appendBytes(fr.read(tempCursorList[i+1]-tempCursorList[i]))
        except OSError as e:
            raise f"{e.strerror} No such file found for loading DiskArray"
        return newD
    
    def __del__(self):
        """At the destruction of this intance it takes sure that no file is left over."""
        logger.debug(f"deleting {self.fileName}")
        if os.path.exists(self.fileName):
            os.remove(self.fileName)