"""
    This script is for logging and saving details of simple scripts
    goals:
        - save the running script itself!
        - save the logging
"""

from tutils import trans_args, trans_init, load_yaml, dump_yaml
import os
import shutil

if __name__ == '__main__':
    args = trans_args()
    logger, config = trans_init(args)
    runs_dir = config['runs_dir']
    file_path = os.path.abspath(__file__)
    shutil.copy(file_path, os.path.join(runs_dir, __file__))

    print(file_path)
    print(__file__)