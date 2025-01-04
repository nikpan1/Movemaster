import torch
import torch.nn as nn


class Interpolate(nn.Module):
    def __init__(self, size, mode: str):
        super(Interpolate, self).__init__()
        self.size = size
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x, size=self.size, mode=self.mode)


class ConvAutoencoder(nn.Module):
    def __init__(self, input_size, layers, grouped, dim: int, input_ch: int = 2, kernel_size: int = 5, kernel_stride: int = 3,
                 return_embeddings: bool = False, decode_only: bool = False):
        super(ConvAutoencoder, self).__init__()

        self.dim = dim
        self.return_embeddings = return_embeddings
        self.decode_only = decode_only
        self.k_s = kernel_stride

        conv_f, conv_t_f, pool_f, unpool_f, out_padding = self._get_layer_functions(dim, kernel_stride)

        # Encoder Layers
        padding = (kernel_size - 1) // 2
        ch_mult = 3 // input_ch

        self.conv1 = conv_f(input_ch, 8 // ch_mult, kernel_size=kernel_size, stride=kernel_stride, padding=padding,
                            groups=grouped[0])
        self.conv2 = conv_f(8 // ch_mult, 12 // ch_mult, kernel_size=kernel_size, stride=kernel_stride, padding=padding,
                            groups=grouped[1])
        self.conv3 = conv_f(12 // ch_mult, 24 // ch_mult, kernel_size=kernel_size, stride=kernel_stride,
                            padding=padding, groups=grouped[2])  # Match channels with deconv1

        self.pool = pool_f(kernel_size=2, stride=2, padding=0, return_indices=True)

        # Decoder Layers
        self.unpool = unpool_f(kernel_size=2, stride=2, padding=0)
        self.deconv1 = conv_t_f(24 // ch_mult, 12 // ch_mult, kernel_size=kernel_size, stride=kernel_stride,
                                padding=padding, output_padding=out_padding[0], groups=grouped[-1])
        self.deconv2 = conv_t_f(12 // ch_mult, 8 // ch_mult, kernel_size=kernel_size, stride=kernel_stride,
                                padding=padding, output_padding=out_padding[1], groups=grouped[-1])
        self.deconv3 = conv_t_f(8 // ch_mult, input_ch, kernel_size=kernel_size, stride=kernel_stride, padding=padding,
                                output_padding=out_padding[2], groups=grouped[-1])

        # Interpolation
        self.interp = Interpolate(input_size, 'linear' if dim == 1 else 'bilinear')

    def _get_layer_functions(self, dim: int, kernel_stride: int):
        if dim == 1:
            conv_f = nn.Conv1d
            conv_t_f = nn.ConvTranspose1d
            pool_f = nn.MaxPool1d
            unpool_f = nn.MaxUnpool1d
            out_padding = [0, 0, 1] if kernel_stride == 2 else [0, 0, 0]
        else:
            conv_f = nn.Conv2d
            conv_t_f = nn.ConvTranspose2d
            pool_f = nn.MaxPool2d
            unpool_f = nn.MaxUnpool2d
            out_padding = [(1, 0), (0, 0), (1, 0)] if kernel_stride[0] == 2 else [(0, 0)] * 3
        return conv_f, conv_t_f, pool_f, unpool_f, out_padding

    def forward(self, x: torch.Tensor, pool_ind: torch.Tensor = None) -> torch.Tensor:
        # Encoder
        if not self.decode_only:
            x = self.conv1(x)
            x = nn.ReLU()(x)

            x = self.conv2(x)
            x = nn.ReLU()(x)

            x = self.conv3(x)
            x = nn.ReLU()(x)

            self.pool_size = x.size()
            x, pool_ind = self.pool(x)
            self.pool_ind = pool_ind

            if self.return_embeddings:
                return x

        # Decoder
        x = self.unpool(x, self.pool_ind, output_size=self.pool_size)

        x = self.deconv1(x)
        x = nn.ReLU()(x)

        x = self.deconv2(x)
        x = nn.ReLU()(x)

        x = self.deconv3(x)

        if self.dim == 1 or self.dim == 2:
            x = self.interp(x)

        return x

    def set_decode_mode(self, val: bool):
        self.decode_only = val
