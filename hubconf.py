"""Entrypoint for ``torch.hub``."""

# TODO how to list dependencies here?

import torch

from deepgazemr import DeepGazeMR as _DeepGazeMR


def DeepGazeMR(pretrained=True):
    model = _DeepGazeMR()

    if pretrained:
        # load checkpoint with model weights
        checkpoint = torch.load('data/deepgazemr-ledov.pt')
        model.load_state_dict(checkpoint['model_state_dict'])

        # load LEDOV center bias
        center_bias = torch.load('data/center-bias-ledov.pt')
        model.center_bias = center_bias

    return model
