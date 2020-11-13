"""Class to test a model"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from rssnet.loaders.dataloaders import MultiFrameCarradaDataset
from rssnet.utils.functions import transform_masks_viz, get_metrics, normalize, define_loss, get_transformations, get_qualitatives
from rssnet.utils.metrics import Evaluator
from rssnet.utils.paths import Paths


class Tester:

    def __init__(self, cfg, vizualizer=None):
        self.cfg = cfg
        self.vizualizer = vizualizer
        self.model = self.cfg['model']
        self.nb_classes = self.cfg['nb_classes']
        self.signal_type = self.cfg['signal_type']
        self.annot_type = self.cfg['annot_type']
        self.process_signal = self.cfg['process_signal']
        self.w_size = self.cfg['w_size']
        self.h_size = self.cfg['h_size']
        self.n_input_ch = self.cfg['nb_input_channels']
        self.batch_size = self.cfg['batch_size']
        self.device = self.cfg['device']
        self.custom_loss = self.cfg['custom_loss']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.paths = Paths().get()
        self.test_results = dict()

    def predict(self, net, seq_loader, iteration=None, get_quali=False):
        net.eval()
        transformations = get_transformations(self.transform_names, split='test',
                                              sizes=(self.w_size, self.h_size))
        criterion = define_loss(self.signal_type, self.custom_loss, self.device)
        running_losses = list()
        metrics = Evaluator(num_class=self.nb_classes)
        if iteration:
            rand_seq = np.random.randint(len(seq_loader))
        with torch.no_grad():
            for i, sequence_data in enumerate(seq_loader):
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
                if iteration and i == rand_seq:
                    rand_frame = np.random.randint(len(frame_dataloader))
                if get_quali:
                    quali_iter = 0
                for j, frame in enumerate(frame_dataloader):
                    data = frame['matrix'].to(self.device).float()
                    mask = frame['mask'].to(self.device).float()
                    data = normalize(data, self.signal_type, self.paths['carrada'],
                                     norm_type=self.norm_type)
                    outputs = net(data).to(self.device)

                    # Rizesing the output to compute real performances
                    if self.signal_type in ('range_angle', 'rdra2ra', 'rad2ra'):
                        if tuple(outputs.shape[2:]) != (256, 256):
                            outputs = F.interpolate(outputs, (256, 256))
                    elif self.signal_type in ('range_doppler', 'rdra2rd', 'rad2rd'):
                        if tuple(outputs.shape[2:]) != (256, 64):
                            outputs = F.interpolate(outputs, (256, 64))
                    else:
                        raise ValueError('Signal type {} is not '
                                         'supported.'.format(self.signal_type))

                    if get_quali:
                        quali_iter = get_qualitatives(outputs, mask, self.paths,
                                                      seq_name, quali_iter)

                    metrics.add_batch(torch.argmax(mask, axis=1).cpu(),
                                      torch.argmax(outputs, axis=1).cpu())
                    loss = criterion(outputs, torch.argmax(mask, axis=1))
                    running_losses.append(loss.data.cpu().numpy()[()])
                    if iteration and i == rand_seq:
                        if j == rand_frame:
                            pred_masks = torch.argmax(outputs, axis=1)[:5]
                            gt_masks = torch.argmax(mask, axis=1)[:5]
                            pred_grid = make_grid(transform_masks_viz(pred_masks, self.nb_classes))
                            gt_grid = make_grid(transform_masks_viz(gt_masks, self.nb_classes))
                            self.vizualizer.update_img_masks(pred_grid, gt_grid, iteration)
            self.test_results = get_metrics(metrics, np.mean(running_losses))
            metrics.reset()
        return self.test_results

    def write_params(self, path):
        with open(path, 'w') as fp:
            json.dump(self.test_results, fp)

    def set_device(self, device):
        self.device = device

    def set_annot_type(self, annot_type):
        self.annot_type = annot_type
