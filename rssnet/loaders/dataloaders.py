"""Classes to load Carrada dataset"""
import os
import numpy as np
from skimage import transform
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from rssnet.loaders.dataset import Carrada
from rssnet.utils.paths import Paths


class SequenceCarradaDataset(Dataset):
    """DataLoader class for Carrada sequences
    Only shuffle sequences
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.seq_names = list(self.dataset.keys())

    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, idx):
        seq_name = self.seq_names[idx]
        return seq_name, self.dataset[seq_name]


class MultiFrameCarradaDataset(Dataset):
    """DataLoader class for Carrada sequences
    Load frames, only for semantic segmentation
    Specific to load several frames (sub sequences)

    PARAMETERS
    ----------
    dataset: SequenceCarradaDataset object
    annotation_type: str
        Supported annotations are 'sparse', 'dense'
    signal_type: str
        Supported signals are 'range_doppler', 'range_angle'
        Comming soon: 'angle_doppler', 'rad'
    path_to_frames: str
        Path to the frames of a given sequence (folder of the sequence)
    process_signal: bool
        Load signal w/ or w/o processing (power, log transform)
    """

    RD_SHAPE = (256, 64)
    RA_SHAPE = (256, 256)
    NB_CLASSES = 4

    def __init__(self, dataset, annotation_type, signal_type, path_to_frames, process_signal,
                 n_frames, transformations=None):
        self.cls = self.__class__
        self.dataset = dataset
        self.annotation_type = annotation_type
        self.signal_type = signal_type
        self.path_to_frames = path_to_frames
        self.process_signal = process_signal
        self.n_frames = n_frames
        self.transformations = transformations
        self.dataset = self.dataset[self.n_frames-1:] # remove n first frames
        self.path_to_annots = os.path.join(self.path_to_frames, 'annotations',
                                           self.annotation_type)

    def transform(self, frame):
        if self.transformations is not None:
            for function in self.transformations:
                frame = function(frame)
        return frame

    def __len__(self):
        """Number of frames per sequence"""
        return len(self.dataset)

    def __getitem__(self, idx):
        init_frame_name = self.dataset[idx][0]
        frame_id = int(init_frame_name)
        frame_names = [str(f_id).zfill(6) for f_id in range(frame_id-self.n_frames+1, frame_id+1)]
        if self.signal_type in ('range_doppler', 'range_angle'):
            matrices = list()
            mask = np.load(os.path.join(self.path_to_annots, init_frame_name, self.signal_type + '.npy'))
        else:
            raise TypeError('Signal type {} is not supported'.format(self.signal_type))
        for frame_name in frame_names:
            if self.signal_type == 'range_doppler':
                if self.process_signal:
                    matrix = np.load(os.path.join(self.path_to_frames, 'range_doppler_processed',
                                                  frame_name + '.npy'))
                else:
                    matrix = np.load(os.path.join(self.path_to_frames, 'range_doppler_raw',
                                                  frame_name + '.npy'))
                matrices.append(matrix)
            elif self.signal_type == 'range_angle':
                if self.process_signal:
                    matrix = np.load(os.path.join(self.path_to_frames, 'range_angle_processed',
                                                  frame_name + '.npy'))
                else:
                    matrix = np.load(os.path.join(self.path_to_frames, 'range_angle_raw',
                                                  frame_name + '.npy'))
                matrices.append(matrix)
            else:
                raise TypeError('Signal type {} is not supported'.format(self.signal_type))

            matrix = np.dstack(matrices)
            matrix = np.rollaxis(matrix, axis=-1)
            frame = {'matrix': matrix, 'mask': mask}
            frame = self.transform(frame)
        return frame


class Rescale:
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, frame):
        matrix, mask = frame['matrix'], frame['mask']
        h, w = matrix.shape[1:]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        # transform.resize induce a smoothing effect on the values
        # transform only the input data
        matrix = transform.resize(matrix, (matrix.shape[0], new_h, new_w))
        return {'matrix': matrix, 'mask': mask}


class Flip:
    """
    Randomly flip the matrix with a proba p
    """

    def __init__(self, proba):
        assert proba <= 1.
        self.proba = proba

    def __call__(self, frame):
        matrix, mask = frame['matrix'], frame['mask']
        h_flip_proba = np.random.uniform(0, 1)
        if h_flip_proba < self.proba:
            matrix = np.flip(matrix, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        v_flip_proba = np.random.uniform(0, 1)
        if v_flip_proba < self.proba:
            matrix = np.flip(matrix, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
        return {'matrix': matrix, 'mask': mask}

def test_sequence():
    """Test for the Sequence class"""
    dataset = Carrada().get('Train')
    dataloader = DataLoader(SequenceCarradaDataset(dataset), batch_size=1,
                            shuffle=False, num_workers=0)
    for i, data in enumerate(dataloader):
        seq_name, seq = data
        assert '000163' in seq.keys()
        assert '000004' in seq['000163'].keys()
        assert 'dense' in seq['000163']['000004']['range_doppler'].keys()
        break


def test_frame_transform():
    """Test for multiframe class + transforms"""
    paths = Paths().get()
    carrada_path = paths['carrada']
    dataset = Carrada(annot_format='light_frame').get('Train')
    seq_dataloader = DataLoader(SequenceCarradaDataset(dataset), batch_size=1,
                                shuffle=True, num_workers=0)
    transformations = [Rescale((400, 1000)), Flip(0.5)]
    for _, data in enumerate(seq_dataloader):
        seq_name, seq = data
        path_to_frames = os.path.join(carrada_path, seq_name[0])
        frame_dataloader = DataLoader(MultiFrameCarradaDataset(seq,
                                                               'dense', 'range_angle',
                                                               path_to_frames,
                                                               process_signal=True,
                                                               n_frames=1,
                                                               transformations=transformations),
                                      shuffle=False,
                                      batch_size=20,
                                      num_workers=0)
        for _, frame in enumerate(frame_dataloader):
            matrix, mask = frame['matrix'], frame['mask']
            assert list(matrix.shape) == [20, 1, 400, 1000]
            assert list(mask.shape) == [20, 4, 256, 256]
            break
        break
        
if __name__ == '__main__':
    test_sequence()
    # test_frame_transform()
