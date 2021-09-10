import argparse
from tutils import trans_args, trans_init

base_config = {}

class MainProcess(object):
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config

    def function1(self):
        pass        

if __name__ == '__main__':    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=int, default=1)
    # parser.add_argument("--func", type=str, default="")
    args = trans_args(parser)
    logger, config = trans_init(args)
    # MainProcess
    mainprocess = MainProcess(logger, config)

    funcname = args.func # or funcname = config['func']
    getattr(mainprocess, funcname)()