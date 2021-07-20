import os
import unittest
import torch
import diskArray
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

    def test_writeTorchTensor(self):
        logger.debug("test writeTorchTensor")
        d = DiskArray(loaderDumper=diskArray.TorchTensorLoaderDumper)
        obj = torch.Tensor([[1., -1.], [1., -1.]])
        obj2 = {'b': 2}
        obj3 = torch.Tensor([[1., -1.], [1., 0]])
        d.append(obj)
        d.append(obj2)
        d.append(obj3)
        self.assertEqual(torch.sum(torch.square(obj - d[0])), 0)
        self.assertEqual(torch.sum(torch.square(obj - d[0])), 0)
        self.assertNotEqual(torch.sum(torch.square(obj - d[2])), 0)

if __name__ == "__main__":
    unittest.main()