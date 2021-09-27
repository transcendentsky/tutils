"""
    This script is for Ablation Study
        that is: train the `same` script with `different` options.

    Used with config file: `ablation.yaml`

    following is the example of `ablation.yaml`

    ###  ablation.yaml  ###
    ablation:
      tag: "ablation_test"
      config_file: "conf.yaml"
      script_file: "demo_train_script.py"
      is: true
      count: 2
      opts:
        training:
          batch_size: [4,8]
          num_workers: [2,3]

    #  opts: ['training:batch_size', 'training:num_workers']

    gpus: "0,1,2,3"
    base_dir: './debug_runs/'
    tag: 'notag'
    #####################
"""
from tutils import load_yaml, dump_yaml
import subprocess
import copy

def _config_parsing(_dict, i):
    # print("_config_parsing debug: ", _dict, i)
    if type(_dict) is dict:
        for key, value in _dict.items():
            _dict[key] = _config_parsing(value, i)
        return _dict
    elif type(_dict) is list:
        return _dict[i]
    else:
        raise ValueError(f"_tmp_get_subconfig Got type error {type(_dict), _dict}")


class AblationTrainer(object):
    # Autorun scripts with ablation params
    def __init__(self, logger, ablation_config):
        self.logger = logger
        self.ablation_config = ablation_config
        self.count_total = ablation_config["ablation"]["count"]
        self.config_file = ablation_config["ablation"]['config_file']
        self.script_file = ablation_config["ablation"]['script_file']
        self.opts = ablation_config['ablation']['opts']
        self.gpus = ablation_config['base']['gpus']
        self.args = ablation_config['ablation']['args']

    def build_tmp_config_file(self):
        base_config = load_yaml(self.config_file)
        print(self.opts)

        # Split ablation opts
        config_list = []
        for i in range(self.count_total):
            opt = copy.deepcopy(self.opts)
            one_opt = _config_parsing(opt, i)
            config = {**base_config, **one_opt}
            config_list.append(config)
        # print("debug build_tmp_config_file: ", config_list)
        return config_list

    def build_tmp_args_list(self):
        # Split ablation opts
        config_list = []
        for i in range(self.count_total):
            opt = copy.deepcopy(self.args)
            one_opt = _config_parsing(opt, i)
            config_list.append(one_opt)
        # print("debug build_tmp_config_file: ", config_list)
        return config_list

    def run(self):
        """
            For Instance, please read run_train / run_test
        """
        pass

    def run_train(self):
        config_list = self.build_tmp_config_file()
        for i, config in enumerate(config_list):
            dump_yaml(self.logger, config, path="tmp/_tmp_config.yaml")
            tag = self.ablation_config['base']['tag'] + "/" + self.ablation_config['ablation']['tag'] + f"_try{i}"
            cmd = f"CUDA_VISIBLE_DEVICES={self.gpus} python {self.script_file} --tag {tag} --config {'tmp/_tmp_config.yaml'}"
            self.logger.info(f"Run cmd: {cmd}")
            ret_value = subprocess.call(cmd, shell=True)
            self.logger.info(f"ret value: {ret_value}")

    def run_test(self):
        config_list = self.build_tmp_config_file()
        for i, config in enumerate(config_list):
            dump_yaml(self.logger, config, path="tmp/_tmp_config.yaml")
            tag = self.ablation_config['base']['tag'] + "/" + self.ablation_config['ablation']['tag'] + f"_try{i}"
            cmd = f"CUDA_VISIBLE_DEVICES={self.gpus} python {self.script_file} --tag {tag} --config {'tmp/_tmp_config.yaml'} --test"
            self.logger.info(f"Run cmd: {cmd}")
            ret_value = subprocess.call(cmd, shell=True)
            self.logger.info(f"ret value: {ret_value}")

    def run_args(self):
        args_list = self.build_tmp_args_list()
        for i, args in enumerate(args_list):
            tag = self.ablation_config['base']['tag'] + "/" + self.ablation_config['ablation']['tag'] + f"_try{i}"
            cmd = f"CUDA_VISIBLE_DEVICES={self.gpus} python {self.script_file} --sessionname {args['sessionname']} -r aae --dataset wflw " \
                  f"--train-count {args['train_count']}"
            self.logger.info(f"Run cmd: {cmd}")
            ret_value = subprocess.call(cmd, shell=True)
            self.logger.info(f"ret value: {ret_value}")


def template(_file_name):
    from tutils import trans_args, trans_init, load_yaml, dump_yaml
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ablation.yaml")
    args = trans_args(parser)
    logger, config = trans_init(args)
    if config['ablation']['is']:
        # Check opts to do ablation
        ablation_trainer = AblationTrainer(logger, config)
        ablation_trainer.train()
        ###
        ablation_trainer.test()

if __name__ == '__main__':
    template(__file__)