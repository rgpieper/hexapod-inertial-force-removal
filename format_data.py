#!/usr/bin/python

import ezc3d
import numpy as np
import pandas as pd
import IPython

class C3DMan:
    def __init__(self, c3d_path: str):

        self.c3d_path = c3d_path
        self.c3d_data = ezc3d.c3d(self.c3d_path)

if __name__ == "__main__":

    TestTrial = C3DMan("data/fullgrid_loaded_01.c3d")
    
    IPython.embed()