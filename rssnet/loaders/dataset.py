"""Class to load the CARRADA dataset"""
import os
import json
from rssnet.utils.paths import Paths


class Carrada:
    """Class to load CARRADA dataset"""

    def __init__(self, annot_format='frame'):
        self.paths = Paths().get()
        self.warehouse = self.paths['warehouse']
        self.carrada = self.paths['carrada']
        if annot_format == 'frame':
            self.annotations = self._load_frame_oriented()
        elif annot_format == 'instance':
            self.annotations = self._load_instance_oriented()
        elif annot_format == 'light_frame':
            self.annotations = self._load_light_frame_oriented()
        elif annot_format == 'selected_light_frame':
            self.annotations = self._load_selected_light_frame_oriented()
        else:
            raise TypeError('Annotation format {} is not supported.'.format(annot_format))
        self.data_seq_ref = self._load_data_seq_ref()
        self.train = dict()
        self.validation = dict()
        self.test = dict()
        self._split()

    def _load_data_seq_ref(self):
        path = os.path.join(self.carrada, 'data_seq_ref.json')
        with open(path, 'r') as fp:
            data_seq_ref = json.load(fp)
        return data_seq_ref

    def _load_frame_oriented(self):
        path = os.path.join(self.carrada, 'annotations_frame_oriented.json')
        with open(path, 'r') as fp:
            annotations = json.load(fp)
        return annotations

    def _load_instance_oriented(self):
        path = os.path.join(self.carrada, 'annotations_instance_oriented.json')
        with open(path, 'r') as fp:
            annotations = json.load(fp)
        return annotations

    def _load_light_frame_oriented(self):
        path = os.path.join(self.carrada, 'light_dataset_frame_oriented.json')
        with open(path, 'r') as fp:
            annotations = json.load(fp)
        return annotations

    def _load_selected_light_frame_oriented(self):
        path = os.path.join(self.carrada, 'selected_light_dataset_frame_oriented.json')
        with open(path, 'r') as fp:
            annotations = json.load(fp)
        return annotations

    def _split(self):
        for sequence in self.annotations.keys():
            split = self.data_seq_ref[sequence]['split']
            if split == 'Train':
                self.train[sequence] = self.annotations[sequence]
            elif split == 'Validation':
                self.validation[sequence] = self.annotations[sequence]
            elif split == 'Test':
                self.test[sequence] = self.annotations[sequence]
            else:
                raise TypeError('Type {} is not supported for splits.'.format(split))

    def get(self, split):
        if split == 'Train':
            return self.train
        if split == 'Validation':
            return self.validation
        if split == 'Test':
            return self.test
        raise TypeError('Type {} is not supported for splits.'.format(split))


def test():
    """Method to test the dataset"""
    annotations = Carrada().get('Train')
    assert '2019-09-16-12-52-12' in annotations.keys()
    assert '000163' in annotations['2019-09-16-12-52-12'].keys()
    assert '000004' in annotations['2019-09-16-12-52-12']['000163'].keys()
    assert 'range_doppler' in annotations['2019-09-16-12-52-12']['000163']['000004'].keys()
    assert 'dense' in annotations['2019-09-16-12-52-12']['000163']['000004']['range_doppler'].keys()


if __name__ == '__main__':
    test()
