# coding: utf-8

import os
import sys
# from tutils import trans_args, trans_init


class Learner(object):
    def __init__(self):
        pass


def train(logger, config, args):
    # datasets
    dataset = DatasetTraining(config, mode="two_images")
    
    model = Learner(config=config, logger=logger)
    trainer = Trainer(config=config, logger=logger)
    trainer.fit(model, dataset, valset=None)


if __name__ == '__main__':
    args = trans_args()
    logger, config = trans_init(args)
    train(logger, config, args)