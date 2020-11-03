"""Class to computes metrics for Carrada"""
import numpy as np
from scipy.stats import hmean
from sklearn.metrics import confusion_matrix


class Evaluator:

    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def get_pixel_acc_class(self, harmonic_mean=False):
        """Pixel Accuracy"""
        acc_by_class = np.diag(self.confusion_matrix) / np.nansum(self.confusion_matrix, axis=1)
        acc_by_class = np.nan_to_num(acc_by_class)
        if harmonic_mean:
            acc = hmean(acc_by_class)
        else:
            acc = np.mean(acc_by_class)
        return acc, acc_by_class

    def get_pixel_recall_class(self, harmonic_mean=False):
        """Pixel Recall"""
        recall_by_class = np.diag(self.confusion_matrix) / np.nansum(self.confusion_matrix, axis=0)
        recall_by_class = np.nan_to_num(recall_by_class)
        if harmonic_mean:
            recall = hmean(recall_by_class)
        else:
            recall = np.mean(recall_by_class)
        return recall, recall_by_class

    def get_miou_class(self, harmonic_mean=False):
        """Mean Intersection over Union"""
        miou_by_class = np.diag(self.confusion_matrix) / (np.nansum(self.confusion_matrix, axis=1)
                                                          + np.nansum(self.confusion_matrix, axis=0)
                                                          - np.diag(self.confusion_matrix))
        miou_by_class = np.nan_to_num(miou_by_class)
        if harmonic_mean:
            miou = hmean(miou_by_class)
        else:
            miou = np.mean(miou_by_class)
        return miou, miou_by_class

    def get_dice_class(self, harmonic_mean=False):
        _, acc_by_class = self.get_pixel_acc_class()
        _, recall_by_class = self.get_pixel_recall_class()
        # Add epsilon term to avoid /0
        dice_by_class = 2*acc_by_class*recall_by_class/(acc_by_class + recall_by_class + 1e-8)
        if harmonic_mean:
            dice = hmean(dice_by_class)
        else:
            dice = np.mean(dice_by_class)
        return dice, dice_by_class

    def _generate_matrix(self, labels, predictions):
        matrix = confusion_matrix(labels.flatten(), predictions.flatten(),
                                  labels=list(range(self.num_class)))
        return matrix

    def add_batch(self, labels, predictions):
        assert labels.shape == predictions.shape
        self.confusion_matrix += self._generate_matrix(labels, predictions)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
