import os
import torch
from models import PatchTST_gc



class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'PatchTST_gc': PatchTST_gc,
        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        pass

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


