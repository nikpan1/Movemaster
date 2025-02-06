import torch
import torch.nn as nn

def make_divisible(v: int, divisor: int) -> int:
    """Round v up to the nearest integer divisible by divisor."""
    return int((v + divisor - 1) // divisor) * divisor

class Interpolate(nn.Module):
    def __init__(self, size, mode: str):
        super(Interpolate, self).__init__()
        self.size = size
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x, size=self.size, mode=self.mode)

class ConvAutoencoder(nn.Module):
    def __init__(self, input_size, layers, grouped, dim: int, input_ch: int, kernel_size: int,
                 kernel_stride: int, return_embeddings: bool, decode_only: bool):
        super(ConvAutoencoder, self).__init__()

        self.dim = dim
        self.return_embeddings = return_embeddings
        self.decode_only = decode_only
        self.k_s = kernel_stride

        conv_f, conv_t_f, pool_f, unpool_f, out_padding = self._get_layer_functions(dim, kernel_stride)
        padding = (kernel_size - 1) // 2

        # Base channel numbers.
        base_ch1, base_ch2, base_ch3 = 8, 12, 24

        # Adjust channels to be divisible by the desired grouping values.
        # Note: grouped is expected to be a sequence (e.g., [group0, group1, group2, group_dec])
        ch1 = make_divisible(base_ch1, grouped[0])
        ch2 = make_divisible(base_ch2, grouped[1])
        ch3 = make_divisible(base_ch3, grouped[2])

        # Determine groups for each conv layer.
        g1 = grouped[0] if (input_ch % grouped[0] == 0 and ch1 % grouped[0] == 0) else 1
        g2 = grouped[1] if (ch1 % grouped[1] == 0 and ch2 % grouped[1] == 0) else 1
        g3 = grouped[2] if (ch2 % grouped[2] == 0 and ch3 % grouped[2] == 0) else 1
        # For the decoder, check all relevant channel dimensions.
        g_dec = grouped[-1] if (ch3 % grouped[-1] == 0 and ch2 % grouped[-1] == 0 and
                                 ch1 % grouped[-1] == 0 and input_ch % grouped[-1] == 0) else 1

        # Encoder Layers
        self.conv1 = conv_f(input_ch, ch1, kernel_size=kernel_size, stride=kernel_stride,
                            padding=padding, groups=g1)
        self.conv2 = conv_f(ch1, ch2, kernel_size=kernel_size, stride=kernel_stride,
                            padding=padding, groups=g2)
        self.conv3 = conv_f(ch2, ch3, kernel_size=kernel_size, stride=kernel_stride,
                            padding=padding, groups=g3)

        self.pool = pool_f(kernel_size=2, stride=2, padding=0, return_indices=True)

        # Decoder Layers
        self.unpool = unpool_f(kernel_size=2, stride=2, padding=0)
        self.deconv1 = conv_t_f(ch3, ch2, kernel_size=kernel_size, stride=kernel_stride,
                                padding=padding, output_padding=out_padding[0],
                                groups=g_dec)
        self.deconv2 = conv_t_f(ch2, ch1, kernel_size=kernel_size, stride=kernel_stride,
                                padding=padding, output_padding=out_padding[1],
                                groups=g_dec)
        self.deconv3 = conv_t_f(ch1, input_ch, kernel_size=kernel_size, stride=kernel_stride,
                                padding=padding, output_padding=out_padding[2],
                                groups=g_dec)

        # Interpolation to recover the desired output size.
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
            # Assuming kernel_stride is a tuple for 2D convolutions.
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
