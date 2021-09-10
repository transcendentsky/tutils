from tutils import trans_args, trans_init
import os
from tutils.tutils.csv_logger import CSVLogger

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
    args = trans_args()
    print(args)
    logger, config = trans_init(args)
    if args.test:
        test(logger, config, args)
    else:
        train(logger, config, args)