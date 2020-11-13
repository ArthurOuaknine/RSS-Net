"""Script to set the path to CARRADA in the config.ini file"""
import os
import sys
import argparse
from rssnet.utils.configurable import Configurable
from rssnet.utils import RSSNET_HOME

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings paths for training and testing.')
    parser.add_argument('--carrada', default='/datasets_local',
                        help='Path to the CARRADA dataset.')
    parser.add_argument('--logs', default='/root/workspace/logs',
                        help='Path to the save the logs and models.')
    args = parser.parse_args()
    configurable = Configurable(os.path.join(RSSNET_HOME, 'config.ini'))
    configurable.set('data', 'warehouse', args.carrada)
    configurable.set('data', 'logs', args.logs)
    with open(os.path.join(RSSNET_HOME, 'config.ini'), 'w') as fp:
        configurable.config.write(fp)
