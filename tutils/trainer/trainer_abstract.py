
import torch
from tutils.tutils.ttimer import tenum, timer
from tutils.tutils import tfilename, CSVLogger, MultiLogger
from torch.cuda.amp import autocast, GradScaler
from tutils.trainer.recorder import Recorder
from .utils.trainer_utils import MultiOptimizer, MultiScheduler, VoidTimer, dict_to_str, _get_time_str



class AbstractTrainer:
    def __init__(self,
                 config,
                 tester,
                 monitor,
                 rank='cuda',
                 world_size=0,
                 ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.tester = tester
        # base
        self.tag = config['base']['tag']
        self.runs_dir = config['base']['runs_dir']
        #
        self.max_epochs = config['training'].get('num_epochs', 400)
        self.batch_size = config["training"].get('batch_size', 4)
        self.num_workers = config['training'].get('num_workers', 0)
        self.save_interval = config['training'].get('save_interval', 50)

        self.load_pretrain_model = config['training'].get('load_pretrain_model', False)
        self.pretrain_model = config['training'].get('pretrain_model', None)
        self.load_optimizer = config['training'].get('load_optimizer', False)
        self.val_check_interval = config['training'].get('val_check_interval', 50)
        self.training_log_interval = config['training'].get('training_log_interval', 1)
        self.use_amp = config['training'].get('use_amp', False)
        self.save_latest_only = config['training'].get('save_latest_only', False)

        if self.rank != 'cuda' and self.world_size > 0:
            self.master_addr = self.ddp_config.get('master_addr', 'localhost')
            self.master_port = str(self.ddp_config.get('master_port', '25700'))
            self.ddp_config = config['training'].get('ddp', dict())
            self.dist_url = 'tcp://' + self.master_addr + ":" + self.master_port
            torch.distributed.init_process_group(backend="nccl", init_method=self.dist_url,
                                                 world_size=self.world_size, rank=self.rank)

        self.logging_available = (self.rank == 0 or self.rank == 'cuda')
        self.trainloader = None
        self.optimizer = None
        self.scheduler = None
        self.init_timers()

        if self.use_amp:
            self.scalar = GradScaler()
            print("Debug settings: use amp=",self.use_amp)

        # Logging, in GPU 0
        self.recorder_mode = config['logger'].get("recorder_reduction", "sum")
        if self.logging_available:
            print("Logger at Process(rank=0)")
            self.recorder = Recorder(reduction=self.recorder_mode)
            self.recorder_test = Recorder(reduction=self.recorder_mode)
            self.logger = None
            self.csvlogger = CSVLogger(tfilename(self.runs_dir, "best_record"))
            self.monitor = monitor
            self.tester = tester

    def init_timers(self):
        self.timer_epoch = timer("one epoch") if self.logging_available else VoidTimer()
        self.timer_batch = timer("a batch") if self.logging_available else VoidTimer()
        self.timer_data = timer("data time") if self.logging_available else VoidTimer()
        self.timer_net = timer("net forwarding") if self.logging_available else VoidTimer()
        self.timer_eval = timer("evaluation") if self.logging_available else VoidTimer()
        self.timer_write = timer("writing files") if self.logging_available else VoidTimer()
        self.timer_5 = timer("others") if self.logging_available else VoidTimer()

    def init_model(self, model, trainset, **kwargs):

        # if self.rank == 0:
        #     assert isinstance(model, LearnerModule), "model type error!"
        #     self.logger = model.configure_logger()['logger']
        #
        # # Initialize Models and DataLoader and Optimizers
        # if self.load_pretrain_model:
        #     model.load()
        # model.net = model.net.to(self.rank)
        # model.net = nn.SyncBatchNorm.convert_sync_batchnorm(model.net)
        # model.net = torch.nn.parallel.DistributedDataParallel(model.net, device_ids=[self.rank], find_unused_parameters=True)
        #
        # sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
        # assert self.batch_size % self.world_size == 0
        # per_device_batch_size = self.batch_size // self.world_size
        # self.trainloader = torch.utils.data.DataLoader(
        #     trainset, batch_size=per_device_batch_size, num_workers=self.num_worker,
        #     pin_memory=True, sampler=sampler, drop_last=True)
        # return model
        raise NotImplementedError


    def fit(self, model, trainset):
        model = self.init_model(model, trainset)
        optimizer, scheduler, start_epoch = self.configure_optim(model)

        for epoch in range(start_epoch, self.max_epochs):
            self.on_before_zero_grad()
            # Training
            self.timer_epoch()
            do_training_log = (epoch % self.training_log_interval == 0)
            self.train(model, self.trainloader, epoch, optimizer, scheduler, do_training_log)

            # epoch logger !
            if do_training_log and self.logging_available:
                _dict = self.recorder.cal_metrics()
                _dict['time_total'] = self.timer_epoch()

                # print(_dict)
                # assert isinstance(lr, float), f"Got lr={lr}, type: {type(lr)}"
                loss_str = ""
                for k, v in _dict.items():
                    loss_str += "{}:{:.4f} ".format(k, v)
                # lr = optimizer.param_groups[0]['lr']
                lr = self.get_lr(optimizer)
                _dict['lr'] = lr
                loss_str += "{}:{:.6e} ".format('lr', lr)
                self.logger.info(f"Epoch {epoch}: {loss_str}")
                self.logger.add_scalars(_dict, step=epoch, tag='train')
                time_log_scalars = self.timer_epoch()
            # Evaluation
            if epoch % self.val_check_interval == 0 and self.logging_available:
                print("Note: Tester runs on <rank 0> only")
                if self.tester is not None:
                    out = self.tester.test(model, epoch, self.rank)
                    if self.monitor is not None:
                        best_dict = self.monitor.record(out, epoch)
                        self.recorder_test.record({**best_dict, **out})
                        if best_dict['isbest']:
                            self.save(model, epoch, type='best')
                            self.csvlogger.record({**best_dict, **out, "time": _get_time_str()})
                        self.logger.info(f"\n[*] {dict_to_str(best_dict)}[*] Epoch {epoch}: \n{dict_to_str(out)}")
                        self.logger.add_scalars(out, step=epoch, tag='test')
                    else:
                        self.logger.info(f"\n[*] Epoch {epoch}: {dict_to_str(out)}")
            time_eval = self.timer_epoch()
            if epoch % self.save_interval == 0 and self.logging_available:
                self.save(model, epoch, 'latest', optimizer)
                time_save_model = self.timer_epoch()
            # dd = {"time_train": _dict['time_total'], "time_log_scalars":time_log_scalars, "time_eval":time_eval, "time_save_model": time_save_model}
            # print(dd)
        print("Training is Over for GPU rank ", self.rank)
        self.cleanup()

    def configure_optim(self, model, **kwargs):
        # Set optimizer and scheduler
        optim_configs = model.configure_optimizers()
        assert isinstance(optim_configs, dict)
        optimizer = optim_configs['optimizer']
        scheduler = optim_configs['scheduler']

        if isinstance(optimizer, list):
            optimizer = MultiOptimizer(optimizer)
        if isinstance(scheduler, list):
            scheduler = MultiScheduler(scheduler)
        if self.load_optimizer:
            start_epoch = model.load_optim(optimizer)
        else:
            start_epoch = 0
        return optimizer, scheduler, start_epoch


    def on_before_zero_grad(self, **kwargs):
        pass

    def train(self, model, trainloader, epoch, optimizer, scheduler=None, do_training_log=True):
        model.train()

        if do_training_log and self.logging_available:
            self.recorder.clear()
            time_record = 0.1111
            self.timer_batch()

        for load_time, batch_idx, data in tenum(trainloader):
            model.on_before_zero_grad()
            optimizer.zero_grad()
            self.timer_data()
            # training steps
            for k, v in data.items():
                if type(v) == torch.Tensor:
                    data[k] = v.to(self.rank)
            time_data_cuda = self.timer_data()
            if self.use_amp:
                with autocast():
                    self.timer_net()
                    out = model.training_step(data, batch_idx)
                    assert isinstance(out, dict)
                    time_fd = self.timer_net()
                    loss = out['loss']
                    self.scalar.scale(loss).backward()
                    self.scalar.step(optimizer)
                    self.scalar.update()
                    time_bp = self.timer_net()
            else:
                self.timer_net()
                out = model.training_step(data, batch_idx)
                if torch.isnan(out['loss']):
                    print("Nan Value: ", out['loss'])
                    raise ValueError
                assert isinstance(out, dict)
                time_fd = self.timer_net()
                loss = out['loss']
                loss.backward()
                optimizer.step()
                time_bp = self.timer_net()

            time_batch = self.timer_batch()
            # batch logger !
            if do_training_log and self.logging_available:
                out['time_load'] = load_time
                out['time_cuda'] = time_data_cuda
                out['time_forward'] = time_fd
                out['time_bp'] = time_bp
                out['time_record'] = time_record
                out['time_batch'] = time_batch
                self.timer_data()
                self.recorder.record(out)
                time_record = self.timer_data()
            # for debug !
            if epoch == 0:
                if self.logging_available:
                    self.logger.info("[*] Debug Checking Pipeline !!!")
                break
        if scheduler is not None:
            scheduler.step()
        self.on_after_training(d=out)

    def on_after_training(self, d, *args, **kwargs):
        pass

    def save(self, model, epoch, type=None, optimizer=None, **kwargs):
        if self.logging_available:
            if type is None:
                if self.save_interval > 0 and epoch % self.save_interval == 0:
                    save_name = "/ckpt/model_epoch_{}.pth".format(epoch)
                    model.save(tfilename(self.runs_dir, save_name), epoch=epoch)
                    self.logger.info(f"Epoch {epoch}: Save model to ``{save_name}``! ")
            elif type == 'best':
                save_name = "/ckpt/best_model_epoch_{}.pth".format(epoch)
                model.save(tfilename(self.runs_dir, save_name), epoch=epoch, is_best=True)
                self.logger.info(f"[Best model] Epoch {epoch}: Save model to ``{save_name}``! ")
            elif type == 'latest':
                if self.save_interval > 0 and epoch % self.save_interval == 0:
                    save_name = "/ckpt/model_latest.pth"
                    model.save(tfilename(self.runs_dir, save_name), epoch=epoch, is_latest=True)
                    save_optim_name = "/ckpt/optim_latest.pth"
                    model.save_optim(tfilename(self.runs_dir, save_optim_name), optimizer=optimizer, epoch=epoch)
                    self.logger.info(f"Epoch {epoch}: Save checkpoint to ``{save_name}``")

    def cleanup(self):
        if self.rank != 'cuda':
            torch.distributed.destroy_process_group()

    def info(self, msg, *args, **kwargs):
        if self.logging_available:
            self.logger.info(msg, *args, **kwargs)

    def get_lr(self, optimizer):
        if isinstance(optimizer, MultiOptimizer):
            return optimizer.get_lr()
        else:
            return optimizer.param_groups[0]['lr']
