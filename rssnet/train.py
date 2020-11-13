"""Main script to train RSS-Net"""
import argparse
import json

from rssnet.learners.initializer import Initializer
from rssnet.learners.model import Model
from rssnet.models.rssnet import RSSNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file.',
                        default='model_configs/rssnet_config.json')
    args = parser.parse_args()
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)

    init = Initializer(cfg)
    data = init.get_data()
    net = RSSNet(nb_classes=data['cfg']['nb_classes'],
                 n_channels=data['cfg']['nb_input_channels'])
    Model(net, data).train()

if __name__ == '__main__':
    main()
