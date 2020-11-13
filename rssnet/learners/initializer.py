"""Class to initialize training pipeline"""
import os
import json
from torch.utils.data import DataLoader

from rssnet.loaders.dataset import Carrada
from rssnet.loaders.dataloaders import SequenceCarradaDataset
from rssnet.utils.paths import Paths


class Initializer:
    """Class to initialize training pipeline

    PARAMETERS
    ----------
    cfg: dict
        Config dict containing the parameters for training
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.paths = Paths().get()

    def _get_data(self):
        data = Carrada(annot_format=self.cfg['annot_format'])
        train = data.get('Train')
        val = data.get('Validation')
        test = data.get('Test')
        return [train, val, test]

    def _get_datasets(self):
        data = self._get_data()
        trainset = SequenceCarradaDataset(data[0])
        valset = SequenceCarradaDataset(data[1])
        testset = SequenceCarradaDataset(data[2])
        return [trainset, valset, testset]

    def _get_dataloaders(self):
        trainset, valset, testset = self._get_datasets()
        trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
        valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
        return [trainloader, valloader, testloader]

    def _structure_data(self):
        data = dict()
        dataloaders = self._get_dataloaders()
        name_exp = (self.cfg['name'] + '_' +
                    'e' + str(self.cfg['nb_epochs']) + '_' +
                    'lr' + str(self.cfg['lr']) + '_' +
                    's' + str(self.cfg['torch_seed']))
        self.cfg['name_exp'] = name_exp
        folder_path = os.path.join(self.paths['logs'], self.cfg['dataset'],
                                   self.cfg['model'], self.cfg['signal_type'],
                                   name_exp)

        temp_folder_path = folder_path + '_' + str(self.cfg['version'])
        while os.path.exists(temp_folder_path):
            self.cfg['version'] += 1
            temp_folder_path = folder_path + '_' + str(self.cfg['version'])
        folder_path = temp_folder_path

        self.paths['results'] = os.path.join(folder_path, 'results')
        self.paths['writer'] = os.path.join(folder_path, 'boards')
        os.makedirs(self.paths['results'], exist_ok=True)
        os.makedirs(self.paths['writer'], exist_ok=True)

        with open(os.path.join(folder_path, 'config.json'), 'w') as fp:
            json.dump(self.cfg, fp)

        data['cfg'] = self.cfg
        data['paths'] = self.paths
        data['dataloaders'] = dataloaders
        return data

    def get_data(self):
        """Return parameters of the training"""
        return self._structure_data()
