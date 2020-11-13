import os
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from rssnet.loaders.dataloaders import Rescale, Flip
from rssnet.utils import RSSNET_HOME


def get_class_weights(path_to_weights, signal_type):
    """Load class weights for custom loss"""
    if signal_type in ('range_angle', 'rdra2ra', 'rad2ra'):
        file_name = 'ra_weights.json'
    elif signal_type in ('range_doppler', 'rdra2rd', 'rad2rd'):
        file_name = 'rd_weights.json'
    else:
        raise ValueError('Signal type {} is not supported.'.format(signal_type))
    with open(os.path.join(path_to_weights, file_name), 'r') as fp:
        weights = json.load(fp)
    weights = np.array([weights['background'], weights['pedestrian'],
                        weights['cyclist'], weights['car']])
    weights = torch.from_numpy(weights)
    return weights


def transform_masks_viz(masks, nb_classes):
    masks = masks.unsqueeze(1)
    masks = (masks.float()/nb_classes)
    return masks


def get_metrics(metrics, loss):
    metrics_values = dict()
    metrics_values['loss'] = loss.item()
    acc, acc_by_class = metrics.get_pixel_acc_class()  # harmonic_mean=True)
    recall, recall_by_class = metrics.get_pixel_recall_class()  # harmonic_mean=True)
    miou, miou_by_class = metrics.get_miou_class()  # harmonic_mean=True)
    dice, dice_by_class = metrics.get_dice_class()
    metrics_values['acc'] = acc
    metrics_values['acc_by_class'] = acc_by_class.tolist()
    metrics_values['recall'] = recall
    metrics_values['recall_by_class'] = recall_by_class.tolist()
    metrics_values['miou'] = miou
    metrics_values['miou_by_class'] = miou_by_class.tolist()
    metrics_values['dice'] = dice
    metrics_values['dice_by_class'] = dice_by_class.tolist()
    return metrics_values


def old_normalize(data, signal_type):
    if signal_type in ('rdra2rd', 'rdra2ra'):
        # Normalize representation independently
        for i in range(data.shape[1]):
            min_value = torch.min(data[:, i, :, :])
            max_value = torch.max(data[:, i, :, :])
            data[:, i, :, :] = torch.div(torch.sub(data[:, i, :, :], min_value),
                                         torch.sub(max_value, min_value))
        return data
    else:
        min_value = torch.min(data)
        max_value = torch.max(data)
        norm_data = torch.div(torch.sub(data, min_value), torch.sub(max_value, min_value))
        return norm_data


def normalize(data, signal_type, carrada_path, norm_type='local'):
    if signal_type in ('range_doppler', 'range_angle', 'rad2rd', 'rad2ra') and \
       norm_type in ('local'):
        min_value = torch.min(data)
        max_value = torch.max(data)
        norm_data = torch.div(torch.sub(data, min_value), torch.sub(max_value, min_value))
        return norm_data

    if signal_type in ('rdra2rd', 'rdra2ra'):
        if norm_type == 'train':
            with open(os.path.join(carrada_path, 'rd_stats.json'), 'r') as fp:
                rd_stats = json.load(fp)
            with open(os.path.join(carrada_path, 'ra_stats.json'), 'r') as fp:
                ra_stats = json.load(fp)
        elif norm_type == 'tvt':
            with open(os.path.join(carrada_path, 'rd_stats_all.json'), 'r') as fp:
                rd_stats = json.load(fp)
            with open(os.path.join(carrada_path, 'ra_stats_all.json'), 'r') as fp:
                ra_stats = json.load(fp)

        # Normalize representation independently
        if norm_type in ('train', 'tvt'):
            for i in range(data.shape[1]):
                if i%2 == 0:
                    # range-Doppler
                    min_value = torch.tensor(rd_stats['min_val'])
                    max_value = torch.tensor(rd_stats['max_val'])
                else:
                    # range-angle
                    min_value = torch.tensor(ra_stats['min_val'])
                    max_value = torch.tensor(ra_stats['max_val'])
                data[:, i, :, :] = torch.div(torch.sub(data[:, i, :, :], min_value),
                                             torch.sub(max_value, min_value))
        elif norm_type in ('local'):
            for i in range(data.shape[1]):
                min_value = torch.min(data[:, i, :, :])
                max_value = torch.max(data[:, i, :, :])
                data[:, i, :, :] = torch.div(torch.sub(data[:, i, :, :], min_value),
                                             torch.sub(max_value, min_value))
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))
        return data

    elif signal_type == 'range_doppler':
        if norm_type == 'train':
            with open(os.path.join(carrada_path, 'rd_stats.json'), 'r') as fp:
                rd_stats = json.load(fp)
        elif norm_type == 'tvt':
            with open(os.path.join(carrada_path, 'rd_stats_all.json'), 'r') as fp:
                rd_stats = json.load(fp)
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))

        min_value = torch.tensor(rd_stats['min_val'])
        max_value = torch.tensor(rd_stats['max_val'])
        norm_data = torch.div(torch.sub(data, min_value),
                              torch.sub(max_value, min_value))
        return norm_data

    elif signal_type == 'range_angle':
        if norm_type == 'train':
            with open(os.path.join(carrada_path, 'ra_stats.json'), 'r') as fp:
                ra_stats = json.load(fp)
        elif norm_type == 'tvt':
            with open(os.path.join(carrada_path, 'ra_stats_all.json'), 'r') as fp:
                ra_stats = json.load(fp)
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))

        min_value = torch.tensor(ra_stats['min_val'])
        max_value = torch.tensor(ra_stats['max_val'])
        norm_data = torch.div(torch.sub(data, min_value),
                              torch.sub(max_value, min_value))
        return norm_data

    elif signal_type in ('rad2rd', 'rad2ra'):
        if norm_type == 'train':
            with open(os.path.join(carrada_path, 'rad_stats.json'), 'r') as fp:
                rad_stats = json.load(fp)
        elif norm_type == 'tvt':
            with open(os.path.join(carrada_path, 'rad_stats_all.json'), 'r') as fp:
                rad_stats = json.load(fp)
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))

        min_value = torch.tensor(rad_stats['min_val'])
        max_value = torch.tensor(rad_stats['max_val'])
        norm_data = torch.div(torch.sub(data, min_value),
                              torch.sub(max_value, min_value))
        return norm_data
    else:
        raise TypeError('Signal {} is not supported.'.format(signal_type))


def define_loss(signal_type, custom_loss, device):
    if custom_loss == 'wce':
        path_to_weights = os.path.join(RSSNET_HOME, 'model_configs')
        weights = get_class_weights(path_to_weights, signal_type)
        loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
    else:
        loss = nn.CrossEntropyLoss()
    return loss


def get_transformations(transform_names, split='train', sizes=None):
    transformations = list()
    if 'rescale' in transform_names:
        transformations.append(Rescale(sizes))
    if 'flip' in transform_names and split == 'train':
        transformations.append(Flip(0.5))
    return transformations


def mask_to_img(mask):
    mask_img = np.zeros((mask.shape[0],
                         mask.shape[1], 3), dtype=np.uint8)
    mask_img[mask == 1] = [255, 0, 0]
    mask_img[mask == 2] = [0, 255, 0]
    mask_img[mask == 3] = [0, 0, 255]
    mask_img = Image.fromarray(mask_img)
    return mask_img


def get_qualitatives(outputs, masks, paths, seq_name, quali_iter):
    folder_path = os.path.join(paths['temp'], seq_name[0])
    os.makedirs(folder_path, exist_ok=True)
    outputs = torch.argmax(outputs, axis=1).cpu().numpy()
    masks = torch.argmax(masks, axis=1).cpu().numpy()
    for i in range(outputs.shape[0]):
        mask_img = mask_to_img(masks[i])
        mask_img.save(os.path.join(folder_path, 'mask_{}.png'.format(quali_iter)))
        output_img = mask_to_img(outputs[i])
        output_img.save(os.path.join(folder_path, 'output_{}.png'.format(quali_iter)))
        quali_iter += 1
    return quali_iter
