"""Class to get global paths"""
import os
from rssnet.utils import RSSNET_HOME
from rssnet.utils.configurable import Configurable


class Paths(Configurable):

    def __init__(self):
        self.config_path = os.path.join(RSSNET_HOME, 'config.ini')
        super().__init__(self.config_path)
        self.paths = dict()
        self._build()

    def _build(self):
        warehouse = self.config['data']['warehouse']
        self.paths['warehouse'] = warehouse
        self.paths['logs'] = self.config['data']['logs']
        self.paths['carrada'] = os.path.join(warehouse, 'Carrada')

    def get(self):
        return self.paths
