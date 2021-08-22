"""
    This script is for logging and saving details of simple scripts
    goals:
        - save the running script itself!
        - save the logging
"""

CONFIG = {
    'base_dir': './runs-simple/'
}

def template(_file_name=__file__):
    from tutils import trans_args, trans_init, load_yaml, dump_yaml, trans_configure
    from tutils.framework import CSVLogger
    import os
    import shutil

    # args = trans_args()
    # logger, config = trans_init(args)
    logger, config = trans_configure(config=CONFIG)
    runs_dir = config['runs_dir']
    file_path = os.path.abspath(_file_name)
    parent, name = os.path.split(_file_name)
    output_path = os.path.join(runs_dir, name)

    print("debug: __file__ ", __file__)
    print("debug: _file_name", _file_name)
    print("debug ", runs_dir)
    print("debug: ", file_path, output_path)
    print("debug ", config)

    shutil.copy(file_path, output_path)
    csv_logger = CSVLogger(runs_dir + "/csv")

    return {"logger": logger, "config":config, "csv_logger": csv_logger}
