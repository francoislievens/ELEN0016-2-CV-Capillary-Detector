"""The implementation of U-Net and FCRN-A models.
We are using an already existing implementation of
the UNet architecture (but we don't use pre-trained
weights.
Source of the code: https://github.com/NeuroSYS-pl/objects_counting_dmap/
"""
from typing import Tuple
import torch
from torch import nn
import os


def conv_block(channels: Tuple[int, int],
               size: Tuple[int, int],
               stride: Tuple[int, int]=(1, 1),
               N: int=1):
    """
    Create a block with N convolutional layers with ReLU activation function.
    The first layer is IN x OUT, and all others - OUT x OUT.

    Args:
        channels: (IN, OUT) - no. of input and output channels
        size: kernel size (fixed for all convolution in a block)
        stride: stride (fixed for all convolution in a block)
        N: no. of convolutional layers

    Returns:
        A sequential container of N convolutional layers.
    """
    # a single convolution + batch normalization + ReLU block
    block = lambda in_channels: nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=channels[1],
                  kernel_size=size,
                  stride=stride,
                  bias=False,
                  padding=(size[0] // 2, size[1] // 2)),
        nn.BatchNorm2d(num_features=channels[1]),
        nn.ReLU()
    )
    # create and return a sequential container of convolutional layers
    # input size = channels[0] for first block and channels[1] for all others
    return nn.Sequential(*[block(channels[bool(i)]) for i in range(N)])


class ConvCat(nn.Module):
    """Convolution with upsampling + concatenate block."""

    def __init__(self,
                 channels: Tuple[int, int],
                 size: Tuple[int, int],
                 stride: Tuple[int, int]=(1, 1),
                 N: int=1):
        """
        Create a sequential container with convolutional block (see conv_block)
        with N convolutional layers and upsampling by factor 2.
        """
        super(ConvCat, self).__init__()
        self.conv = nn.Sequential(
            conv_block(channels, size, stride, N),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, to_conv: torch.Tensor, to_cat: torch.Tensor):
        """Forward pass.

        Args:
            to_conv: input passed to convolutional block and upsampling
            to_cat: input concatenated with the output of a conv block
        """
        return torch.cat([self.conv(to_conv), to_cat], dim=1)


class UNet(nn.Module):
    """
    U-Net implementation.

    Ref. O. Ronneberger et al. "U-net: Convolutional networks for biomedical
    image segmentation."
    """

    def __init__(self, filters: int=64, input_filters: int=3, **kwargs):
        """
        Create U-Net model with:

            * fixed kernel size = (3, 3)
            * fixed max pooling kernel size = (2, 2) and upsampling factor = 2
            * fixed no. of convolutional layers per block = 2 (see conv_block)
            * constant no. of filters for convolutional layers

        Args:
            filters: no. of filters for convolutional layers
            input_filters: no. of input channels
        """
        super(UNet, self).__init__()
        # first block channels size
        initial_filters = (input_filters, filters)
        # channels size for downsampling
        down_filters = (filters, filters)
        # channels size for upsampling (input doubled because of concatenate)
        up_filters = (2 * filters, filters)

        # downsampling
        self.block1 = conv_block(channels=initial_filters, size=(3, 3), N=2)
        self.block2 = conv_block(channels=down_filters, size=(3, 3), N=2)
        self.block3 = conv_block(channels=down_filters, size=(3, 3), N=2)

        # upsampling
        self.block4 = ConvCat(channels=down_filters, size=(3, 3), N=2)
        self.block5 = ConvCat(channels=up_filters, size=(3, 3), N=2)
        self.block6 = ConvCat(channels=up_filters, size=(3, 3), N=2)

        # density prediction
        self.block7 = conv_block(channels=up_filters, size=(3, 3), N=2)
        self.density_pred = nn.Conv2d(in_channels=filters, out_channels=1,
                                      kernel_size=(1, 1), bias=False)

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        # Reshape the inputs
        input = input.reshape(-1, 1, 240, 240).float()
        # use the same max pooling kernel size (2, 2) across the network
        pool = nn.MaxPool2d(2)

        # downsampling
        block1 = self.block1(input)
        pool1 = pool(block1)
        block2 = self.block2(pool1)
        pool2 = pool(block2)
        block3 = self.block3(pool2)
        pool3 = pool(block3)

        # upsampling
        block4 = self.block4(pool3, block3)
        block5 = self.block5(block4, block2)
        block6 = self.block6(block5, block1)

        # density prediction
        block7 = self.block7(block6)
        return self.density_pred(block7)


def load_prev_unet(model, optimizer, name, path, device='cpu'):

    starting_epoch = 0

    # Directory management
    if not os.path.exists('{}/{}'.format(path, name)):
        os.mkdir('{}/{}'.format(path, name))
    if not os.path.exists('{}/{}/model_weights.pt'.format(path, name)):
        # Reset logs files:
        f = open('{}/{}/logs.csv'.format(path, name), 'w')
        f.write('epoch,train_loss,test_loss\n')
        f.close()
        return model, optimizer, starting_epoch

    else:
        try:
            # Load logs file to determine the epoch index
            logs = pd.read_csv('{}/{}/logs.csv'.format(path, name), sep=',')
            starting_epoch = np.max(logs['epoch'].to_numpy()) + 1

            # Load model and optimizer weights
            model.load_state_dict(torch.load('{}/{}/model_weights.pt'.format(path, name), map_location=device))
            optimizer.load_state_dict(torch.load('{}/{}/optimizer_weights.pt'.format(path, name), map_location=device))
        except:
            print('ERROR: Impossible to load previous model')
            sys.exit(1)

        return model, optimizer, starting_epoch

def load_UNet(model, name, path, device):

    try:
        model.load_state_dict(torch.load('{}/{}/model_weights.pt'.format(path, name), map_location=device))
    except:
        print('ERROR: impossible to load the model at path {}/{}'.format(path, name))
        sys.exit(1)

# --- PYTESTS --- #

def run_network(input_channels: int):
    """Generate a random image, run through network, and check output size."""
    sample = torch.ones((1, input_channels, 224, 224))
    result = UNet(input_filters=input_channels)(sample)
    assert result.shape == (1, 1, 224, 224)


def test_UNet_color():
    """Test U-Net on RGB images."""
    run_network(3)


def test_UNet_grayscale():
    """Test U-Net on grayscale images."""
    run_network(1)


