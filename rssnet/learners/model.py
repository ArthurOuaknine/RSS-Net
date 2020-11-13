"""Class to train a PyTorch model"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from rssnet.utils.functions import normalize, define_loss, get_transformations
from rssnet.utils.tensorboard_visualizer import TensorboardVisualizer
from rssnet.loaders.dataloaders import MultiFrameCarradaDataset
from rssnet.learners.tester import Tester


class Model(nn.Module):

    def __init__(self, net, data):
        super().__init__()
        self.net = net
        self.cfg = data['cfg']
        self.paths = data['paths']
        self.dataloaders = data['dataloaders']
        self.signal_type = self.cfg['signal_type']
        self.process_signal = self.cfg['process_signal']
        self.annot_type = self.cfg['annot_type']
        self.w_size = self.cfg['w_size']
        self.h_size = self.cfg['h_size']
        self.batch_size = self.cfg['batch_size']
        self.nb_epochs = self.cfg['nb_epochs']
        self.lr = self.cfg['lr']
        self.lr_step = self.cfg['lr_step']
        self.loss_step = self.cfg['loss_step']
        self.val_step = self.cfg['val_step']
        self.viz_step = self.cfg['viz_step']
        self.torch_seed = self.cfg['torch_seed']
        self.numpy_seed = self.cfg['numpy_seed']
        self.nb_classes = self.cfg['nb_classes']
        self.device = self.cfg['device']
        self.custom_loss = self.cfg['custom_loss']
        self.comments = self.cfg['comments']
        self.n_input_ch = self.cfg['nb_input_channels']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.writer = SummaryWriter(self.paths['writer'])
        self.visualizer = TensorboardVisualizer(self.writer)
        self.tester = Tester(self.cfg, self.visualizer)
        self.results = dict()

    def train(self):
        self.writer.add_text('Comments', self.comments)
        train_loader, val_loader, test_loader = self.dataloaders
        transformations = get_transformations(self.transform_names,
                                              sizes=(self.w_size, self.h_size))
        self._set_seeds()
        self.net.apply(self._init_weights)
        running_losses = list()
        criterion = define_loss(self.signal_type, self.custom_loss, self.device)
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        iteration = 0
        best_val_acc = 0
        self.net.to(self.device)

        for epoch in range(self.nb_epochs):
            if epoch % self.lr_step == 0 and epoch != 0:
                scheduler.step()
            for _, sequence_data in enumerate(train_loader):
                seq_name, seq = sequence_data
                path_to_frames = os.path.join(self.paths['carrada'], seq_name[0])
                frame_dataloader = DataLoader(MultiFrameCarradaDataset(seq,
                                                                       self.annot_type,
                                                                       self.signal_type,
                                                                       path_to_frames,
                                                                       self.process_signal,
                                                                       self.n_input_ch,
                                                                       transformations),
                                              shuffle=False,
                                              batch_size=self.batch_size,
                                              num_workers=4)
                for _, frame in enumerate(frame_dataloader):
                    data = frame['matrix'].to(self.device).float()
                    mask = frame['mask'].to(self.device).float()
                    data = normalize(data, self.signal_type, self.paths['carrada'],
                                     norm_type=self.norm_type)
                    optimizer.zero_grad()
                    outputs = self.net(data).to(self.device)
                    mask = F.interpolate(mask, (self.w_size, self.h_size))
                    loss = criterion(outputs, torch.argmax(mask, axis=1))
                    loss.backward()
                    optimizer.step()
                    running_losses.append(loss.data.cpu().numpy()[()])
                    if iteration % self.loss_step == 0:
                        train_loss = np.mean(running_losses)
                        print('[Epoch {}/{}, iter {}]: '
                              'train loss {}'.format(epoch+1,
                                                     self.nb_epochs,
                                                     iteration,
                                                     train_loss))
                        self.visualizer.update_train_loss(train_loss, iteration)
                        running_losses = list()
                        self.visualizer.update_learning_rate(scheduler.get_lr()[0], iteration)
                    if iteration % self.val_step == 0 and iteration > 0:
                        if iteration % self.viz_step == 0 and iteration > 0:
                            val_metrics = self.tester.predict(self.net, val_loader, iteration)
                        else:
                            val_metrics = self.tester.predict(self.net, val_loader)
                        self.visualizer.update_val_metrics(val_metrics, iteration)
                        print('[Epoch {}/{}] Validation loss: {}'.format(epoch+1,
                                                                         self.nb_epochs,
                                                                         val_metrics['loss']))
                        print('[Epoch {}/{}] Validation Pixel Acc: {}'.format(epoch+1,
                                                                              self.nb_epochs,
                                                                              val_metrics['acc']))
                        print('[Epoch {}/{}] Validation Pixel Acc by class: '
                              '{}'.format(epoch+1,
                                          self.nb_epochs,
                                          val_metrics['acc_by_class']))

                        if val_metrics['acc'] > best_val_acc and iteration > 0:
                            best_val_acc = val_metrics['acc']
                            test_metrics = self.tester.predict(self.net, test_loader)
                            print('[Epoch {}/{}] Test loss: {}'.format(epoch+1,
                                                                       self.nb_epochs,
                                                                       test_metrics['loss']))
                            print('[Epoch {}/{}] Test Pixel Acc: {}'.format(epoch+1,
                                                                            self.nb_epochs,
                                                                            test_metrics['acc']))
                            print('[Epoch {}/{}] Test Pixel Acc by class: '
                                  '{}'.format(epoch+1,
                                              self.nb_epochs,
                                              test_metrics['acc_by_class']))

                            self.results['train_loss'] = train_loss.item()
                            self.results['val_metrics'] = val_metrics
                            self.results['test_metrics'] = test_metrics
                            self._save_results()
                        self.net.train()  # Train mode after evaluation process
                    iteration += 1
        self.writer.close()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, 0., 1.)
                nn.init.constant_(m.bias, 0.)

    def _save_results(self):
        with open(os.path.join(self.paths['results'], 'results.json'), "w") as fp:
            json.dump(self.results, fp)
        torch.save(self.net.state_dict(),
                   os.path.join(self.paths['results'],
                                'model.pt'))

    def _set_seeds(self):
        torch.cuda.manual_seed_all(self.torch_seed)
        torch.manual_seed(self.torch_seed)
        np.random.seed(self.numpy_seed)
