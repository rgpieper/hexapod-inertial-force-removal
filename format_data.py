#!/usr/bin/python

import ezc3d
import numpy as np
import pandas as pd
import IPython

# NOTE: Nexus graph depicts first analog datapoint to align with frame 1 and datapoint 5 aligning with frame 2

class C3DMan:
    def __init__(self, c3d_path: str):

        self.c3d_path = c3d_path
        self.c3d_data = ezc3d.c3d(self.c3d_path)

        desc_analog = self.c3d_data["parameters"]["ANALOG"]["DESCRIPTIONS"]["value"]
        lab_analog = self.c3d_data["parameters"]["ANALOG"]["LABELS"]["value"]
        self.analogs_idx = {det: i for i, det in enumerate(list(zip(desc_analog,lab_analog)))}

        desc_point = self.c3d_data["parameters"]["POINT"]["DESCRIPTIONS"]["value"]
        lab_point = self.c3d_data["parameters"]["POINT"]["LABELS"]["value"]
        self.points_idx = {det: i for i, det in enumerate(list(zip(desc_point,lab_point)))}

        self.accelerations = pd.DataFrame()
        self.forces = pd.DataFrame()

        IPython.embed()

    def extract_accelerometers(
            self,
            desc_labs: dict[str,tuple[str,str]] = {
                'a1x': ('Analog Accelerometer::Acceleration [19,1]', 'Acceleration.X'),
                'a1y': ('Analog Accelerometer::Acceleration [19,2]', 'Acceleration.Y'),
                'a1z': ('Analog Accelerometer::Acceleration [19,3]', 'Acceleration.Z'),
                'a2x': ('Analog Accelerometer::Acceleration [20,1]', 'Acceleration.X'),
                'a2y': ('Analog Accelerometer::Acceleration [20,2]', 'Acceleration.Y'),
                'a2z': ('Analog Accelerometer::Acceleration [20,3]', 'Acceleration.Z'),
                'a3x': ('Analog Accelerometer::Acceleration [21,1]', 'Acceleration.X'),
                'a3y': ('Analog Accelerometer::Acceleration [21,2]', 'Acceleration.Y'),
                'a3z': ('Analog Accelerometer::Acceleration [21,3]', 'Acceleration.Z'),
                'a4x': ('Analog Accelerometer::Acceleration [22,1]', 'Acceleration.X'),
                'a4y': ('Analog Accelerometer::Acceleration [22,2]', 'Acceleration.Y'),
                'a4z': ('Analog Accelerometer::Acceleration [22,3]', 'Acceleration.Z')
            }
    ):
        
        acc_idx = [self.analogs_idx[q] for q in desc_labs.values()]
        acc_data = self.c3d_data["data"]["analogs"][0][acc_idx]

        self.accelerations[list(desc_labs.keys())] = acc_data.T

    def extract_forceplate(
            self,
            desc_labs: dict[str,tuple[str,str]] = {
                'fx12': ('Kistler Force Plate 2.0.0.0::Raw [27]', 'Raw.FX12'),
                'fx34': ('Kistler Force Plate 2.0.0.0::Raw [27]', 'Raw.FX34'),
                'fy14': ('Kistler Force Plate 2.0.0.0::Raw [27]', 'Raw.FY14'),
                'fy23': ('Kistler Force Plate 2.0.0.0::Raw [27]', 'Raw.FY23'),
                'fz1': ('Kistler Force Plate 2.0.0.0::Raw [27]', 'Raw.FZ1'),
                'fz2': ('Kistler Force Plate 2.0.0.0::Raw [27]', 'Raw.FZ2'),
                'fz3': ('Kistler Force Plate 2.0.0.0::Raw [27]', 'Raw.FZ3'),
                'fz4': ('Kistler Force Plate 2.0.0.0::Raw [27]', 'Raw.FZ4')
            }
    ):
        
        force_idx = [self.analogs_idx[q] for q in desc_labs.values()]
        force_data = self.c3d_data["data"]["analogs"][0][force_idx]

        self.forces[list(desc_labs.keys())] = force_data.T

if __name__ == "__main__":

    TestTrial = C3DMan("data/fullgrid_loaded_01.c3d")
    TestTrial.extract_accelerometers()