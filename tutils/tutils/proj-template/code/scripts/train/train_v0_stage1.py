"""
    should consist in path of saving dirs
"""


if __name__ == '__main__':
    args = trans_args()
    logger, config = trans_init(args)
    
    print(config['extag'])
    # keep base_dir same
    print(config['base_dir'])
    # keep runs_dir = base_dir + tag + extag (stage1)
    print(config['runs_dir'])
    
    train(logger, config, args)