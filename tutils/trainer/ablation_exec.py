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
import pandas as pd
import csv
from tutils.tutils import tfilename, CSVLogger
import os


# def _config_parsing(_dict, i):
#     # print("_config_parsing debug: ", _dict, i)
#     if type(_dict) is dict:
#         for key, value in _dict.items():
#             ret = _config_parsing(value, i)
#             if ret is None:
#                 return None
#             _dict[key] = ret
#         return _dict
#     elif type(_dict) is list:
#         if len(_dict) < i:
#             return _dict[i]
#         else:
#             return None
#     else:
#         raise ValueError(f"_tmp_get_subconfig Got type error {type(_dict), _dict}")

def _config_parsing(d, i):
    _d = copy.deepcopy(d)
    # print("_config_parsing debug: ", _d, i)
    if type(_d) is dict:
        for key, value in _d.items():
            _d[key] = _config_parsing(value, i)
        return _d
    elif type(_d) is list:
        return _d[i]
    else:
        raise ValueError(f"_tmp_get_subconfig Got type error {type(_d), _d}")
        
def parse_opts(d):
    r = []
    i = 0
    while True:
        try:
            r.append(_config_parsing(d, i))
        except:
            print("End parsing")
            break
        i += 1
    print("Parsing Opts length: ", len(r))
    return r

class AblationTrainer(object):
    # Autorun scripts with ablation params
    def __init__(self, logger, ablation_config):
        print("\n[*] Starting AblationTrainer !\n")
        self.logger = logger
        self.ablation_config = ablation_config
        self.tag = ablation_config['base']['tag']
        self.runs_dir = ablation_config['base']['runs_dir']

        self.gather_record = ablation_config['ablation'].get('gather_record', False)
        self.config_file = ablation_config["ablation"].get('config_file', None)
        # self.script_file = ablation_config["ablation"]['script_file']
        self.opts = ablation_config['ablation']['opts']
        self.running_cmd = ablation_config['ablation']['running_cmd']
        self.fixed_opts = ablation_config['ablation'].get('fixed_opts', {})

        self.parse_mode = ablation_config['ablation'].get('parse_mode', "configs")
        if self.parse_mode == "configs":
            assert self.config_file is not None
            self.config_list, self.params_list = self.build_tmp_config_file()
        elif self.parse_mode == "args":
            self.config_list, self.params_list = self.build_tmp_args_list()
        else:
            raise TypeError("Mode 'configs' and 'args' are supported ONLY ! ")

        self.config_len = len(self.config_list)

        self.abla_tags = ablation_config['ablation'].get("tags", None)
        self.auto_tag = True if self.abla_tags is not None else False
        if isinstance(self.abla_tags, list):
            print(f"[Ablation Trainer] tag_type: List")
            assert len(self.abla_tags) >= self.config_len, f"Error! tags is not enough, len(self.abla_tags) = {len(self.abla_tags)} < len(self.config_list) = {self.config_len}"
        elif isinstance(self.abla_tags, str):
            print("[Ablation Trainer] tag_type: str")
            if self.abla_tags == "auto":
                self.abla_tags = [f"{self.tag}/try{i}" for i in range(self.config_len)]
                print("Auto tags : ", self.abla_tags)
            else:
                raise NotImplementedError
        elif self.abla_tags is None:
            print(" No abla_tags ")

        self.csvlogger = CSVLogger(self.runs_dir)
        # import ipdb; ipdb.set_trace()

    def build_tmp_config_file(self):
        base_config = load_yaml(self.config_file)

        # Split ablation opts
        params_list = parse_opts(self.opts)
        config_list = []
        for params in params_list:
            config = {**base_config, **self.fixed_opts, **params}
            config_list.append(config)
        # import ipdb; ipdb.set_trace()
        return config_list, params_list

    def build_tmp_args_list(self):
        # Split ablation opts
        params_list = parse_opts(self.opts)
        config_list = []
        for params in params_list:
            one = {**self.fixed_opts, **params}
            config_list.append(one)
        return config_list, params_list

    def run(self):
        self.logger.info(f"Running parse mode: {self.parse_mode}")
        if self.parse_mode == "configs":
            self.run_with_config()
        elif self.parse_mode == "args":
            self.run_with_args()
        else:
            raise TypeError

    def run_with_config(self):
        assert len(self.config_list) > 0, f"Config File Error: Got NO opts to parse"
        for i in range(self.config_len):
            config = self.config_list[i]
            params = self.params_list[i]
            dump_yaml(self.logger, config, path=tfilename(self.runs_dir, "tmp/_tmp_config.yaml"))
            cmd = self.running_cmd + f" --config {tfilename(self.runs_dir, 'tmp/_tmp_config.yaml')}"

            if self.auto_tag:
                tag = self.abla_tags[i]
                cmd += f" --tag {tag}"
            else: tag = None

            self.logger.info(f"Run cmd: {cmd}")
            p = subprocess.Popen(cmd, shell=True)
            p.wait()
            if self.gather_record:
                self.integrate_records(config, {**params}, tag)

    def run_with_args(self):
        assert len(self.config_list) > 0, f"Config File Error: Got NO opts to parse"
        for i in range(self.config_len):
            args = self.config_list[i]
            params = self.params_list[i]
            s = ""
            for k, v in args.items():
                s += f" --{k} {v}"
            cmd = self.running_cmd + s
            
            if self.auto_tag:
                tag = self.abla_tags[i]
                cmd += f" --tag {tag}"
            else: tag = None

            self.logger.info(f"Run cmd: {cmd}")
            p = subprocess.Popen(cmd, shell=True)
            p.wait()

            if self.gather_record:
                raise NotImplementedError("Gather_record Only supported when parse_mode='configs'. ")
                self.integrate_records(args,  {**params}, tag)

    def integrate_records(self, config, params, tag):
        runs_dir = self._search_runs_dir(config, tag)
        record_path = tfilename(runs_dir, "best_record", "record.csv")
        if record_path is None:
            print("[Warning] No cfg of 'runs_dir' ")
            return
        record = self._read_records(record_path)
        if record is None:            
            print(f"[Warning] No record file in `{record_path}` ")
            return
        assert isinstance(record, dict)
        record = {**record, **flatten_dict(params)}
        self.csvlogger.record(record)

    def _search_runs_dir(self, config, tag):
        assert isinstance(config, dict), f"config is not dict ?, Got {config} "
        d = config.get("runs_dir", None)
        if d is not None and d != '':
            return d
        d = config['base'].get("runs_dir", None)
        if d is not None and d != '':
            return d
        # raise FileNotFoundError
        # construct runs_dir path according to initilizer
        print("[Ablation Trainer] guess the runs_dir: ", end='')
        experiment = config['base'].get('experiment', '')
        stage = config['base'].get('stage', '')
        _tag = config['base'].get('tag', '')
        _tag = _tag if _tag != '' else tag
        runs_dir = os.path.join(config['base']['base_dir'], experiment, _tag, stage)
        print(runs_dir)
        return runs_dir

    def _read_records(self, record_pth):
        data = pd.read_csv(record_pth)
        _d = data[-1:].to_dict(orient='list')
        _r = dict()
        for k, v in _d.items():
            _r[k] = v[-1]
        return _r


def flatten_dict(d, parent_name=None):
    """
    flatten dict: 
    config={
        'base':
            'experiment': 'test',
    }
        ==> 
    config={
        'base.experiment': 'test',
    }
    """
    s = parent_name + "." if parent_name is not None else ""
    if isinstance(d, dict):
        _d = dict()
        for k, v in d.items():
            if not isinstance(v, dict):
                _d = {**_d, **{s+k: v}}
            else:
                _d = {**_d, **flatten_dict(d[k], s + k)}
        return _d


def execute():
    from tutils import trans_args, trans_init, load_yaml, dump_yaml
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ablation.yaml")
    args = trans_args(parser)
    logger, config = trans_init(args)
    ablation_trainer = AblationTrainer(logger, config)
    ablation_trainer.run()

if __name__ == '__main__':
    execute()