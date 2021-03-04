import yaml
import numpy as np

class TParams(object):
    def __init__(self):
        # common hyper-params
        self.batch_size = 16
        self.learning_rate = 1