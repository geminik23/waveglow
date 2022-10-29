import time
import torch
import numpy as np
import tqdm 
import pandas as pd
import os
from utils import to_device

from tqdm.autonotebook import tqdm as nb_tqdm
from tqdm import tqdm


class GeneralTrainer(object):
    def __init__(self, 
                 model, 
                 optimizer_builder, 
                 loss_func, 
                 lr_schedule_builder=None, 
                 score_metric:dict={}, 
                 pre_net=None,
                 checkpoint_dir="model_cp"):
        self.model = model
        self.optimizer_builder = optimizer_builder
        self.loss_func = loss_func
        self.score_metric = score_metric
        self.lr_schedule_builder = lr_schedule_builder
        self.pre_net = pre_net
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.tqdm = tqdm
        self.reset()
    
    def set_tqdm_for_notebook(self, notebook_tqdm=False):
        self.tqdm = nb_tqdm if notebook_tqdm else tqdm
    
    def _init(self, val, test):
        self.result = {}

        record_keys = ["epoch", "total time", "train loss"]
        if val: record_keys.append("val loss")
        if test: record_keys.append("test loss")
        for eval_score in self.score_metric:
            record_keys.append("train " + eval_score )
            if val: record_keys.append("val " + eval_score )
            if test: record_keys.append("test "+ eval_score )
            
        for item in record_keys:
            self.result[item] = []

    def reset(self):
        self.total_time = 0
        self.last_epoch = 0
        self.optimizer = self.optimizer_builder(self.model)
        self.lr_schedule = None if self.lr_schedule_builder is None else self.lr_schedule_builder()
        self.result = {}
    
    def load_data(self, filepath):
        self.reset()

        data = torch.load(filepath)
        if data.get('epoch') is not None:
            self.last_epoch = data.get('epoch') 
        if data.get('result') is not None:
            self.result = data.get('result')
        if self.result.get('total time') is not None and len(self.result['total time'])!=0:
            self.total_time = self.result['total time'][-1]
        self.model.load_state_dict(data.get('model_state_dict'))
        self.optimizer.load_state_dict(data.get('optimizer_state_dict'))
        if self.lr_schedule is not None:
            self.lr_schedule.load_state_dict(data.get('rl_schedule_state_dict'))


    def save_data(self, filename):
        torch.save({
            'epoch': self.last_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'rl_schedule_state_dict': None if self.lr_schedule is None else self.rl_schedule.state_dict(),
            'result' : self.result,
            }, os.path.join(self.checkpoint_dir, filename))

    def get_result(self):
        return pd.DataFrame.from_dict(self.result)

    def run_epoch(self, data_loader, device, desc=None, prefix=""):
        losses = []
        ys = [] 
        y_preds= []

        # measure the time
        start = time.time()
        for inputs, labels in self.tqdm(data_loader, desc=desc, leave=False):
            inputs = to_device(inputs, device)
            labels = to_device(labels, device)

            if self.pre_net is not None:
                inputs = self.pre_net(inputs)

            y_hat = self.model(inputs)
            
            loss = self.loss_func(y_hat, labels)


            # only when training
            if self.model.training:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            losses.append(loss.item())

            if len(self.score_metric) > 0 and isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
                y_hat = y_hat.detach().cpu().numpy()

                ys.extend(labels.tolist())
                y_preds.extend(y_hat.tolist())
        
        end = time.time() 

        y_preds = np.asarray(y_preds)

        # for classification problem
        if len(y_preds.shape) == 2 and y_preds.shape[1] > 1: 
            y_preds = np.argmax(y_preds, axis=1)
        
        self.result[prefix + " loss"].append(np.mean(losses))

        for n, score_func in self.score_metric.items():
            try:
                self.result[prefix + " " + n].append( score_func(ys, y_preds) )
            except:
                self.result[prefix + " " + n].append(float("NaN"))
        del y_preds
        del ys

        return end-start


    def train(self, train_loader, val_loader, test_loader=None, 
              epochs=10, device='cpu', reset=True, cp_filename=None, cp_period=10, print_progress=False):
        # initialize

        # init result
        if reset:
            self.reset()

        if len(self.result) == 0 or reset:
            self._init(val_loader is not None, test_loader is not None)


        # set device
        is_cuda = False
        if type(device) == torch.device:
            is_cuda = device.type.startswith('cuda')
        elif type(device) == str:
            is_cuda = device.startswith('cuda')
        
        if is_cuda:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            
        if self.pre_net is not None:
            self.pre_net.to(device)
        
        self.model.to(device)


        for epoch in self.tqdm(range(self.last_epoch + 1, self.last_epoch + 1 + epochs), desc="Epoch"):
            self.model = self.model.train()

            self.total_time += self.run_epoch(train_loader, device, prefix="train", desc="training")
            
            self.result["epoch"].append( epoch )
            self.result["total time"].append( self.total_time )
        
            if val_loader is not None:
                self.model = self.model.eval()
                with torch.no_grad():
                    self.run_epoch(val_loader, device, prefix="val", desc="validating")
                    
            if test_loader is not None:
                self.model = self.model.eval() 
                with torch.no_grad():
                    self.run_epoch(test_loader, device, prefix="test", desc="Testing")
                    
            if self.lr_schedule is not None:
                self.lr_schedule.step()

            self.last_epoch = epoch

            if print_progress:
                total_secs = int(self.total_time)
                print(f"Epoch {epoch} - loss : {self.result['train loss'][-1]}, val_loss : {'None' if val_loader is None else self.result['val loss'][-1]}, time : {total_secs//60}:{total_secs%60}")
            
            if cp_filename is not None and epoch%cp_period == 0:
                self.save_data(cp_filename.format(epoch))
            

        return self.get_result()
