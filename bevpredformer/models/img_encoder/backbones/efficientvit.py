import timm

from collections import OrderedDict
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from bevpredformer.models.img_encoder.backbones.common import Backbone

class EfficientVit(Backbone):
    def __init__(self, checkpoint_path=None,
                 version="efficientvit_l2.r384_in1k", downsample=8):
        super().__init__()
        self.version = version
        self.downsample = downsample
        
        assert downsample == 8, "Currently only supported for downsample 8"

        # Only the first three stages are consumed by the neck. Building the last
        # one leaves it without gradient, which DDP reports as an unused parameter.
        self.model = timm.create_model(version, pretrained=True, features_only=True,
                                       out_indices=(0, 1, 2))
        message = f"EfficientVit exists and is loaded at version {version}"
        self._print_loaded_file(message)


    @rank_zero_only
    def _print_loaded_file(self, message):
        print("# -------- Backbone -------- #")
        print(message, end="\n")

    def forward(self, x, return_all=False):
        endpoints = dict()

        res = self.model(x)
        for i, feat in enumerate(res):
            endpoints[f"reduction_{i + 1}"] = feat

        if not return_all:
            list_keys =  ["reduction_2", "reduction_3"]
        else:
            list_keys = list(endpoints.keys())

        return OrderedDict({f"out{i}": endpoints[k] for i, k in enumerate(list_keys)})
