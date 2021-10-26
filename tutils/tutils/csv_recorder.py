import shutil
import xlwt
from typing import List
import numpy as np
from .tutils import tfilename
from datetime import datetime
import os
import csv
import pandas as pd


def _get_time_str():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class CSVLogger(object):
    """
        record dict:
            record(d={"a": 'a'})
    """
    def __init__(self, logdir, name="record.csv", mode="w"):
        super().__init__() 
        self.logdir = logdir
        self.fname = name
        self._keys_write_state_ = False
        self.filename = tfilename(self.logdir, name)
        self.mode = mode
        
        if mode == "w":
            if os.path.isfile(self.filename):
                backup_name = self.filename.replace(".csv", "."+_get_time_str()+".csv")
                shutil.move(self.filename, backup_name)
                print(f"[CSVLogger] backup excel file `{self.filename}` to `{backup_name}`")
        elif mode == "a+":
            print("[CSVLogger] Add data on existing CSV file!")
            # self.previous_data = self.read_previous(logdir + "/record.csv")
            self._keys_write_state_ = True
        else:
            raise NotImplementedError
 
    def record(self, d):
        assert type(d) == dict, f"Got {type(d)}"
        # Add record time
        d['record_time'] = _get_time_str()
        if not self._keys_write_state_:
            self.write_row(list(d.keys()))
            self._keys_write_state_ = True
        self.write_row(list(d.values()))

    def write_row(self, row):
        with open(self.filename, "a+") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(row)

    def read_previous(self, path):
        data = pd.read_csv(path)
        return data


    # Not needed ! Cool !
    # def standardize(self, v):
    #     if type(v) in [str, int, float]:
    #         return v
    #     elif type(v) == np.ndarray:
    #         return float(v.astype(float))
    #     elif type(v) == np.float:
    #         return np.asscalar(v)
    #     else:
    #         return np.asscalar(v)
    #         # raise NotImplementedError(f"Got {type(v)}")


class ExcelLogger(object):
    def __init__(self, logdir, name="record.xls"):
        self.logdir = logdir
        self.fname = name
        self.col_index = 0
        # States
        self._keys_write_state_ = False
        self.filename = tfilename(self.logdir, name)
        # Backup previous file
        if os.path.isfile(self.filename):
            backup_name = self.filename.replace(".xls", "."+_get_time_str()+".xls")
            shutil.move(self.filename, backup_name)
            print(f"[ExcelLogger] backup excel file `{self.filename}` to `{backup_name}`")

        self.file = xlwt.Workbook(encoding='utf-8')
        self.table = self.file.add_sheet('data')

    def reset(self):
        self._keys_write_state_ = False
        self.col_index = 0

    def record(self, d):
        assert type(d) == dict, f"Got {type(d)}"
        if not self._keys_write_state_:
            self.write_row(list(d.keys()))
            self._keys_write_state_ = True
        self.write_row(list(d.values()))

    def write_row(self, row:list) -> None:
        """ write one row each time, for multiple ops
            Open sheet -> write data -> save sheet
        """
        assert type(row) == list , f"Got {type(row)}"
        i = self.col_index
        for j, v in enumerate(row):
            # standardize types
            v = self.standardize(v)
            # print("debug ", i, j, v, type(v))
            self.table.write(i, j, v)
        self.col_index += 1
        self.file.save(self.filename)

    def standardize(self, v):
        if type(v) in [str, int, float]:
            return v
        elif type(v) == np.ndarray:
            return float(v.astype(float))
        elif type(v) == np.float:
            return np.asscalar(v)
        else:
            return np.asscalar(v)
            # raise NotImplementedError(f"Got {type(v)}")

    def save(self):
        """ save file after all writings, comes with self.write_row() """
        filename = self.fname
        filename = tfilename(self.logdir, filename)
        self.file.save(filename)

    def write_table(self, data:List[dict]) -> None:
        """ Write all data at a time
            data: list [dict]
                [   {'epoch': 0, "time":"2020-7-7", "tag":"tag1",
                        'loss': 1.1, 'loss1':0.3},
                    {'epoch': 1, "time":"2020-7-7", "tag":"tag1",
                        'loss': losses[0], 'loss1':0.3},
                    ...      ]
            Final Output Table should be:
            | epoch | time | loss | loss1 | loss 2| ...
            | 0     | xxxx | 0.1  | 0.2   | 0.23  | ...
            | 1     | xxxx | 0.1  | 0.2   | 0.23  | ...
        """
        self.reset()
        for i in range(len(data)):
            self.record(data[i])
        self.save()


def _get_time_str():
    return datetime.now().strftime('%m%d-%H%M%S')
