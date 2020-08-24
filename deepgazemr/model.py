"""PyTorch implentation of DeepGazeMR."""

from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DeepGazeMR(nn.Module):
    def __init__(self, center_bias=None):
        super().__init__()
        self.features = _Backbone()
        self.readout = _Readout([32, 32, 1])
        self.finalizer = _Finalizer()
        self.center_bias = center_bias
        self._window_length = 16

    def forward(self, clip, center_bias=None):
        center_bias = self._adapt_center_bias(clip.shape[-2:], center_bias)
        center_bias = center_bias.to(clip.device)

        features = self.features(clip)
        priority_map = self.readout(features)
        prediction = self.finalizer(priority_map, center_bias)
        return prediction

    def predict(self, video, center_bias=None):
        center_bias = self._adapt_center_bias(video.shape[-2:], center_bias)
        center_bias = center_bias.to(video.device)

        buffer = None
        buffer_index = 0

        for i in range(video.shape[0]):
            if i < self._window_length - 1:
                yield None
                continue

            elif buffer is None:
                window = video[0:self._window_length]
                buffer = self.features(window)

            else:
                frame = video[i:i+1]
                buffer[buffer_index] = self.features(frame)
                buffer_index = (buffer_index + 1) % self._window_length

            # we assume that the readout network averages features over time,
            # so the frames don't need to be in order
            priority_map = self.readout(buffer)
            prediction = self.finalizer(priority_map, center_bias)
            yield prediction

    def _adapt_center_bias(self, size, center_bias=None):
        center_bias = center_bias if center_bias is not None \
                      else self.center_bias
        assert center_bias is not None, "No center bias given."

        center_bias = center_bias.unsqueeze(0).unsqueeze(0)

        # resize the center bias to the target size and renormalize if
        # necessary
        if center_bias.size()[-1] != size[-1] or \
           center_bias.size()[-2] != size[-2]:
            center_bias = F.interpolate(center_bias, size)
            center_bias /= center_bias.sum()

        return center_bias.log()


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        # load frozen, pretrained VGG19
        self.vgg = models.vgg19(pretrained=True)
        for param in self.vgg.parameters():
            param.requires_grad = False

        # remove the final pooling & FC layers
        self.vgg.features[-1] = nn.Identity()
        self.vgg.avgpool = nn.Identity()
        self.vgg.classifier = nn.Identity()

        self._mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        self._std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

    def forward(self, clip):
        clip = (clip - self._mean.to(clip.device)) / self._std.to(clip.device)
        return self.vgg.features(clip)


class _Readout(nn.Sequential):
    def __init__(self, num_channels, last_norm=False, last_nlin=False):
        num_layers = len(num_channels)
        num_channels = [512] + num_channels  # backbone: 512 feature maps

        layers = OrderedDict()

        for i in range(1, num_layers+1):
            layers[f'conv{i}'] = \
                nn.Conv2d(num_channels[i-1], num_channels[i], 1)

            if last_norm or i < num_layers:
                layers[f'norm{i}'] = \
                    nn.GroupNorm(1, num_channels[i], affine=True)

            if last_nlin or i < num_layers:
                layers[f'nlin{i}'] = nn.Softplus()

        super().__init__(layers)

    def forward(self, features):
        feature_means = features.mean(dim=0, keepdim=True)
        return super().forward(feature_means)


class _Finalizer(nn.Module):
    def __init__(self,
                 sigma=16.0,
                 kernel_size=None,
                 learn_sigma=False,
                 center_bias_weight=1.0,
                 learn_center_bias_weight=True):

        super().__init__()

        if kernel_size is None:
            kernel_size = int(2 * math.ceil(3 * sigma) + 1)

        self.gauss_x = GaussianFilter1d(1, sigma, kernel_size=kernel_size)
        self.gauss_y = GaussianFilter1d(0, sigma, kernel_size=kernel_size)

        if learn_sigma:
            self.gauss_x.sigma.requires_grad = True
            self.gauss_y.sigma.requires_grad = True

        self.center_bias_weight = nn.Parameter(
            torch.tensor([center_bias_weight]),
            requires_grad=learn_center_bias_weight
        )

    def forward(self, priority_map, center_bias):
        # resize to size of the center bias (= output size)
        out = F.interpolate(
            priority_map, size=center_bias.size()[-2:], mode='nearest')

        # apply gaussian filter
        out = self.gauss_x(out)
        out = self.gauss_y(out)

        # add center bias
        out = out + self.center_bias_weight * center_bias

        # remove batch and channel dimension
        out = out.squeeze()

        # normalize
        # REVIEW rather use softmax? -> don't return *log* prediction
        out = out - out.logsumexp(dim=(0, 1))

        return out


class GaussianFilter1d(nn.Module):
    """Differentiable gaussian filter."""

    def __init__(self,
                 dim,
                 sigma,
                 input_dims=2,
                 truncate=4,
                 kernel_size=None,
                 padding_mode='replicate',
                 padding_value=0.0):
        """
        Initialize the Gaussian filter.

        :param dim: the dimension to which the gaussian filter is applied,
            ignoring batch and channel dimension. This does not support
            negative values.
        :param sigma: standard deviation of the gaussian filter (blur size).
        :param input_dims: number of input dimensions ignoring the batch and
            channel dimension, i.e. use input_dims=2 for images (default: 2).
        :param truncate: truncate the filter at this many standard deviations
            (default: 4.0). This has no effect if the ``kernel_size`` is set
            explicitely.
        :param kernel_size: size of the gaussian kernel convolved with the
            input.
        :param padding_mode: padding mode supported by
            ``torch.nn.functional.pad`` (default: 'replicate').
        :param padding_value: value used for constant padding.
        """
        # IDEA determine input_dims dynamically for every input
        super().__init__()

        self.dim = dim
        self.sigma = nn.Parameter(torch.Tensor([sigma]), requires_grad=False)

        # use `kernel_size` if given, otherwise truncate kernel at the given
        # multiply of sigma
        self.kernel_size = \
            kernel_size if kernel_size is not None \
            else 2 * math.ceil(truncate * sigma) + 1

        # create a grid [-n, ..., n] of length kernel_size
        # the grid is transformed into a gaussian kernel only in the forward
        # pass, this way the module is differentiable w.r.t. to sigma
        mean = (self.kernel_size - 1) / 2
        grid = torch.arange(float(self.kernel_size)) - mean

        # reshape the grid so that it can be used as a kernel for F.conv1d
        kernel_shape = [1] * (2 + input_dims)
        kernel_shape[2 + dim] = self.kernel_size
        grid = grid.view(kernel_shape)

        # necessary for moving to GPU when calling .cuda()
        self.register_buffer('grid', grid)

        # setup padding
        self.padding_mode = padding_mode
        self.padding_value = padding_value

        # use asymmetric padding if necessary (even kernel size)
        self.padding = [0] * (2 * input_dims)
        self.padding[dim * 2 + 1] = math.ceil((self.kernel_size - 1) / 2)
        self.padding[dim * 2] = math.ceil((self.kernel_size - 1) / 2)
        self.padding = tuple(reversed(self.padding))

    def forward(self, x):
        """Apply the Gaussian filter to the given tensor."""
        x = F.pad(x, self.padding, self.padding_mode, self.padding_value)

        # create gaussian kernel from grid using current sigma
        kernel = torch.exp(-0.5 * (self.grid / self.sigma) ** 2)
        kernel = kernel / kernel.sum()

        # convolve input with gaussian kernel
        return F.conv1d(x, kernel)
