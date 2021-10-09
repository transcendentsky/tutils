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
            ret = _config_parsing(value, i)
            if ret is None:
                return None
            _dict[key] = ret
        return _dict
    elif type(_dict) is list:
        if len(_dict) < i:
            return _dict[i]
        else:
            return None
    else:
        raise ValueError(f"_tmp_get_subconfig Got type error {type(_dict), _dict}")


class AblationTrainer(object):
    # Autorun scripts with ablation params
    def __init__(self, logger, ablation_config):
        self.logger = logger
        self.ablation_config = ablation_config
        self.config_file = ablation_config["ablation"]['config_file']
        self.script_file = ablation_config["ablation"]['script_file']
        self.opts = ablation_config['ablation']['opts']
        self.running_cmd = ablation_config['ablation']['running_cmd']
        # self.args = ablation_config['ablation']['args']
        self.fixed_opts = ablation_config['ablation']['fixed_opts']

        self.parse_mode = ablation_config['base']['parse_mode']
        if self.parse_mode == "configs":
            self.config_list = self.build_tmp_config_file()
        elif self.parse_mode == "args":
            self.config_list = self.build_tmp_args_list()
        else:
            raise TypeError("Mode 'configs' and 'args' are supported ONLY ! ")

        self.config_len = len(self.config_list)

        self.tags = ablation_config['ablation'].get("tags", None)
        if isinstance(self.tags, list):
            print(f"[Ablation Trainer] {'list'}")
        elif isinstance(self.tags, str):
            print("[Ablation Trainer] ...str")
            if self.tags == "index":
                self.tags = [f"abla_try{i}" for i in range(self.config_len)]
            else:
                raise NotImplementedError
        elif self.tags is None:
            print(" No tags ")

    def run(self):
        if self.parse_mode == "configs":
            self.run_with_config()
        elif self.parse_mode == "args":
            self.run_with_args()
        else:
            raise TypeError

    def run_with_config(self):
        config_list = self.build_tmp_config_file()
        for i, config in enumerate(config_list):
            dump_yaml(self.logger, config, path="tmp/_tmp_config.yaml")
            cmd = self.running_cmd + f" --config {'tmp/_tmp_config.yaml'}"
            runs_dir = self._search_runs_dir(config)
            self.logger.info(f"Run cmd: {cmd}")
            ret_value = subprocess.call(cmd, shell=True)
            self.logger.info(f"ret value: {ret_value}")

    def run_with_args(self):
        args_list = self.build_tmp_args_list()
        for i, args in enumerate(args_list):
            s = ""
            for k, v in args.items():
                s += f" --{k} {v}"
            cmd = self.running_cmd + s
            self.logger.info(f"Run cmd: {cmd}")
            ret_value = subprocess.call(cmd, shell=True)
            self.logger.info(f"ret value: {ret_value}")

    def build_tmp_config_file(self):
        base_config = load_yaml(self.config_file)
        print(self.opts)
        # Split ablation opts
        config_list = []
        for i in range(99999):
            opt = copy.deepcopy(self.opts)
            one_opt = _config_parsing(opt, i)
            if one_opt is None:
                break
            config = {**base_config, **self.fixed_opts, **one_opt}
            config_list.append(config)
        # print("debug build_tmp_config_file: ", config_list)
        return config_list

    def build_tmp_args_list(self):
        # Split ablation opts
        config_list = []
        for i in range(99999):
            opt = copy.deepcopy(self.opts)
            one_opt = _config_parsing(opt, i)
            if one_opt is None:
                break
            one = {**self.fixed_opts, **one}
            config_list.append(opt)
        # print("debug build_tmp_config_file: ", config_list)
        return config_list

    def _search_runs_dir(self, config):
        assert isinstance(config, dict), f"config is not dict ?, Got {config} "
        d = config.get("runs_dir", None)
        if d is not None:
            return d
        d = config['base'].get("runs_dir", None)
        if d is not None:
            return d
        raise FileNotFoundError

    def _get_record_from_csv(self, csv_path):
        

def execute():
    from tutils import trans_args, trans_init, load_yaml, dump_yaml
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ablation.yaml")
    args = trans_args(parser)
    logger, config = trans_init(args)

    ablation_trainer = AblationTrainer(logger, config)
    ablation_trainer.train()
        ###
    ablation_trainer.test()

if __name__ == '__main__':
    execute()