import os
import unittest

from diskArray import DiskArray
import loggers

logger = loggers.getLogger("DiskArrayTest", debug=True)

_current_path = os.path.dirname(os.path.realpath(__file__))

class DiskArrayTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(DiskArrayTest, self).__init__(*args, **kwargs)
    def test_creation(self):
        logger.debug("test creation")
        d = DiskArray()
    # def test_read(self):
    #     logger.debug("test read")
    def test_write(self):
        logger.debug("test write")
        d = DiskArray()
        obj = {'a': 1, 'b': 2}
        d.append(obj)
        self.assertEqual(obj, d[0])
        self.assertEqual(obj, d[0])

if __name__ == "__main__":
    unittest.main()