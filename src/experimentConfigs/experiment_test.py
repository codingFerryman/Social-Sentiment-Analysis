import unittest
import experiment
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import inputFunctions

class ExperimentTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ExperimentTest, self).__init__(*args, **kwargs)
    
    def test_robertaDefault(self):
        pass

if __name__ == "__main__":
    unittest.main()