from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, save_result
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.metrics import metric
from utils.tools import get_model_config
from datetime import datetime
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
from tqdm import tqdm
import numpy as np
import pandas
np.set_printoptions(precision=4)

warnings.filterwarnings('ignore')

class Exp_Convention_Filtering(Exp_Basic):
    def __init__(self, args):
        super(Exp_Convention_Filtering, self).__init__(args)

    def _build_model(self):
        # Set device and print function
        if self.args.use_multi_gpu:
            self.device = self.args.accelerator.device
            self.print = self.args.accelerator.print
        else:
            self.device = self.args.gpu
            self.print = print

        # Retrieve model configurations
        self.args.obs_dim, self.args.state_dim = get_model_config(self.args.model_id)

        # Check for invalid model_id
        if self.args.obs_dim is None or self.args.state_dim is None:
            raise ValueError(f"Invalid model_id: {self.args.model_id}")

        # Initialize the model
        model = self.model_dict[self.args.model].Model(self.args).to(self.device)
        
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
                self.print(n, p.dtype, p.shape)
        model_optim = optim.AdamW([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')         
          
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion(self.args.loss)
        if self.args.use_multi_gpu:
            train_loader, vali_loader, self.model, model_optim, scheduler = self.args.accelerator.prepare(train_loader, vali_loader, self.model, model_optim, scheduler)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, None, None)
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, None)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                
                if (i + 1) % 100 == 0:
                    self.print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_multi_gpu:
                    self.args.accelerator.backward(loss)
                    model_optim.step()
                elif self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            self.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, criterion)
            test_loss = vali_loss
            self.print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                self.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + f'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path), strict=False)
        return self.model

    def vali(self, train_loader, vali_loader, criterion):
        x, x_mask, y, y_mask = vali_loader.dataset.last_insample_window()
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x_mask = torch.tensor(x_mask, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            l, B, C = y.shape
            outputs = torch.zeros((l, B, C)).float().to(self.device)
            id_list = np.arange(0, B, int(self.args.window_length/2))
            id_list = np.append(id_list, B)
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    for i in range(len(id_list) - 1):
                        outputs[:,id_list[i]:id_list[i + 1], :] = self.model(x[:,id_list[i]:id_list[i + 1]],x_mask[:,id_list[i]:id_list[i + 1]],None, None)

            else:
                for i in range(len(id_list) - 1):
                    outputs[:,id_list[i]:id_list[i + 1], :] = self.model(x[:,id_list[i]:id_list[i + 1]],x_mask[:,id_list[i]:id_list[i + 1]],None, None)
            
            true = torch.from_numpy(np.array(y)).to(self.device)
            loss = criterion(outputs, true)

        self.model.train()
        return loss

    def test(self, setting, test=0):
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        x, x_mask, y, y_mask = test_loader.dataset.last_insample_window()
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x_mask = torch.tensor(x_mask, dtype=torch.float32).to(self.device)

        if test:
            self.print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name

            self.print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            load_item = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)


        self.model.eval()

        token_length = int(self.args.window_length/2)
        with torch.no_grad():
            l, B, C = y.shape
            outputs = torch.zeros((l, B, C)).float().to(self.device)
            id_list = np.arange(0, B, token_length)
            id_list = np.append(id_list, B)
            start_time = time.time()
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    for i in range(len(id_list) - 1):
                        outputs[:,id_list[i]:id_list[i + 1], :] = self.model(x[:,id_list[i]:id_list[i + 1]],x_mask[:,id_list[i]:id_list[i + 1]],None, None)

            else:
                for i in range(len(id_list) - 1):
                    outputs[:,id_list[i]:id_list[i + 1], :] = self.model(x[:,id_list[i]:id_list[i + 1]],x_mask[:,id_list[i]:id_list[i + 1]],None, None)
            end_time = time.time()

            outputs = outputs[0, token_length:, :]
            preds = outputs.detach().cpu().numpy()
            trues = y[0, token_length:, :]
            
            x = x.detach().cpu().numpy()
            obs = x[0, token_length:, :]
            
        
        # # save
        # if self.args.save:
        #     save_result(self.args, obs, preds, trues)
        
        inference_time = (end_time - start_time) / (B - token_length) 
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        self.print('mse:{:.4f}, mae:{:.4f},rmse:{:.4f}'.format(mse, mae, rmse))
        f = open("result_long_term_filtering.txt", 'a')
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+ "  \n")
        f.write(setting + "  \n")
        f.write('mse:{:.4f}, mae:{:.4f},rmse:{:.4f}'.format(mse, mae, rmse))
        f.write(f"  Inference time: {inference_time * 1000:.5f} ms")
        f.write('\n')
        f.write('\n')
        f.close()
        print(f"Inference time: {inference_time * 1000:.5f} ms")
        return
