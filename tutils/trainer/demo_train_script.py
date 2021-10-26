

class MainProcess(object):
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config


def train(logger, config, args):
    pass
    # Learner
    # Trainer
    # Trainer.fit(Learner)
    logger.info("code test for demo_train_script.py")
    file_path = os.path.abspath(os.path.dirname(__file__))
    logger.info(file_path)
    logger.info(config['runs_dir'])
    logger.info(config['base_dir'])

def test(logger, config, args):
    logger.info("code test for demo_train_script.py")
    file_path = os.path.abspath(os.path.dirname(__file__))
    logger.info(file_path)
    logger.info(config['runs_dir'])
    logger.info(config['base_dir'])
    pass


if __name__ == '__main__':
    from tutils import trans_args, trans_init
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=int, default=1)
    parser.add_argument("--func", type=str, default="")
    args = trans_args(parser)
    print(args)
    logger, config = trans_init(args)
    
    mainprocess = MainProcess(logger, config)
    funcname = args.func # or funcname = config['func']
    getattr(mainprocess, funcname)()