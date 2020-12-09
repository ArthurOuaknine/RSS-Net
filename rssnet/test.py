"""Script to test a pretrained model"""
import argparse
import os
import json
import torch
from torch.utils.data import DataLoader

from rssnet.utils.paths import Paths
from rssnet.learners.tester import Tester
from rssnet.models.rssnet import RSSNet
from rssnet.loaders.dataset import Carrada
from rssnet.loaders.dataloaders import SequenceCarradaDataset

def test_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file of the model to test.',
                        default='config.json')
    args = parser.parse_args()
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)

    paths = Paths().get()
    path = os.path.join(paths['logs'], cfg['dataset'], cfg['model'],
                        cfg['signal_type'], cfg['name_exp'] + '_' + str(cfg['version']))
    model_path = os.path.join(path, 'results', 'model.pt')
    test_results_path = os.path.join(path, 'results', 'test_results.json')

    model = RSSNet(nb_classes=cfg['nb_classes'], n_channels=cfg['nb_input_channels'])
    model.load_state_dict(torch.load(model_path))
    model.cuda()

    tester = Tester(cfg)
    data = Carrada(annot_format='light_frame')
    test = data.get('Test')
    testset = SequenceCarradaDataset(test)
    seq_testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    tester.set_annot_type(cfg['annot_type'])
    test_results = tester.predict(model, seq_testloader, get_quali=True)
    tester.write_params(test_results_path)

if __name__ == '__main__':
    test_model()
