import timm

from collections import OrderedDict
from pytorch_lightning.utilities import rank_zero_only

from bevpredformer.models.img_encoder.backbones.common import Backbone

class EfficientVit(Backbone):
    def __init__(self, checkpoint_path=None,
                 version="efficientvit_l2.r384_in1k", downsample=8):
        super().__init__()
        self.version = version
        self.downsample = downsample
        
        assert downsample == 8, "Currently only supported for downsample 8"

        self.model = timm.create_model(version, pretrained=True, features_only=True)
        message = f"EfficientVit exists and is loaded at version {version}"
        self._print_loaded_file(message)


    @rank_zero_only
    def _print_loaded_file(self, message):
        print("# -------- Backbone -------- #")
        print(message, end="\n")

    def forward(self, x, return_all=False):
        endpoints = dict()

        res = self.model(x)
        endpoints["reduction_1"] = res[0]
        endpoints["reduction_2"] = res[1]
        endpoints["reduction_3"] = res[2]
        endpoints["reduction_4"] = res[3]

        if not return_all:
            list_keys =  ["reduction_2", "reduction_3"]
        else:
            list_keys = ["reduction_1", "reduction_2", "reduction_3", "reduction_4"]

        return OrderedDict({f"out{i}": endpoints[k] for i, k in enumerate(list_keys)})
