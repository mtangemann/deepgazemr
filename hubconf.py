"""Entrypoint for ``torch.hub``."""

# TODO how to list dependencies here?

from pathlib import Path

import torch

from deepgazemr import DeepGazeMR as _DeepGazeMR

REPOSITORY_PATH = Path(__file__).parent
CHECKPOINT_PATH = REPOSITORY_PATH / 'data' / 'deepgazemr-ledov.pt'
CENTER_BIAS_PATH = REPOSITORY_PATH / 'data' / 'center-bias-ledov.pt'


def DeepGazeMR(pretrained=True):
    model = _DeepGazeMR()

    if pretrained:
        # load checkpoint with model weights
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])

        # load LEDOV center bias
        center_bias = torch.load(CENTER_BIAS_PATH)
        model.center_bias = center_bias

    return model
