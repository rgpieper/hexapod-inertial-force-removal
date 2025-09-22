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

        self.analogs_idx = {description: i for i, description in enumerate(self.c3d_data["parameters"]["ANALOG"]["DESCRIPTIONS"]["value"])}
        self.points_idx = {label: i for i, label in enumerate(self.c3d_data["parameters"]["POINT"]["LABELS"]["value"])}

        self.accelerations = pd.DataFrame()

    def extract_accelerometers(
            self,
            descriptions: dict[str,str] = {
                'a1x': 'Analog Accelerometer::Acceleration [19,1]',
                'a1y': 'Analog Accelerometer::Acceleration [19,2]',
                'a1z': 'Analog Accelerometer::Acceleration [19,3]',
                'a2x': 'Analog Accelerometer::Acceleration [20,1]',
                'a2y': 'Analog Accelerometer::Acceleration [20,2]',
                'a2z': 'Analog Accelerometer::Acceleration [20,3]',
                'a3x': 'Analog Accelerometer::Acceleration [21,1]',
                'a3y': 'Analog Accelerometer::Acceleration [21,2]',
                'a3z': 'Analog Accelerometer::Acceleration [21,3]',
                'a4x': 'Analog Accelerometer::Acceleration [22,1]',
                'a4y': 'Analog Accelerometer::Acceleration [22,2]',
                'a4z': 'Analog Accelerometer::Acceleration [22,3]'
            }
    ):
        
        acc_idx = [self.analogs_idx[q] for q in descriptions.values()]
        acc_data = self.c3d_data["data"]["analogs"][0][acc_idx]

        self.accelerations[descriptions.keys()] = acc_data.T

        IPython.embed()


if __name__ == "__main__":

    TestTrial = C3DMan("data/fullgrid_loaded_01.c3d")
    TestTrial.extract_accelerometers()