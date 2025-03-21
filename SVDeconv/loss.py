from typing import TYPE_CHECKING
from collections import namedtuple
import math
import torch
import torch.nn as nn

from torchvision import models
import numbers
from utils.contextual_loss import calculate_CX_Loss

import torch.nn.functional as F

if TYPE_CHECKING:
    from utils.typing_alias import *


def label_like(label: int, x: "Tensor") -> "Tensor":
    assert label == 0 or label == 1
    v = torch.zeros_like(x) if label == 0 else torch.ones_like(x)
    v = v.to(x.device)
    return v


def soft_zeros_like(x: "Tensor") -> "Tensor":
    zeros = label_like(0, x)
    return torch.rand_like(zeros)


def soft_ones_like(x: "Tensor") -> "Tensor":
    ones = label_like(1, x)
    return ones * 0.7 + torch.rand_like(ones) * 0.5


def zeros_like(x: "Tensor") -> "Tensor":
    zeros = label_like(0, x)
    return zeros


def ones_like(x: "Tensor") -> "Tensor":
    ones = label_like(1, x)
    return ones

def split_tensor_into_grid(tensor, grid):
    """
    分割一个形状为(B, C, H, W)的PyTorch 张量到一个大小为(grid x grid)的网格中。
    每个网格都是一个较小的张量，所有这些张量放在一个列表中返回。
    """
    B, C, H, W = tensor.shape
    grid_size = int(grid ** 0.5) # grid是完全平方数，取平方根得到每边的切分数量
    height_per_grid = H // grid_size
    width_per_grid = W // grid_size
    
    # 初始化结果列表
    images = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            # 计算每个小格的起始和结束索引
            start_i = i * height_per_grid
            end_i = start_i + height_per_grid
            start_j = j * width_per_grid
            end_j = start_j + width_per_grid
            
            # 从原张量中切割小张量
            grid_tensor = tensor[:, :, start_i:end_i, start_j:end_j]
            images.append(grid_tensor)
    
    return images

class GLoss(nn.modules.Module):
    def __init__(self, args):
        super(GLoss, self).__init__()
        self.args = args

        if self.args.lambda_perception or self.args.lambda_contextual:
            self.LossNetwork = Vgg16FeatureExtractor()
            self.LossNetwork.eval()

        self.is_multi = self.args.multi > 1

    def _perception_metric(self, X, Y):
        feat_X = self.LossNetwork(X)
        feat_Y = self.LossNetwork(Y)

        loss = F.mse_loss(feat_X.relu2_2, feat_Y.relu2_2)
        loss = loss + F.mse_loss(feat_X.relu4_3, feat_Y.relu4_3)

        return loss

    def _contextual_metric(self, X, Y):
        feat_X = self.LossNetwork(X)
        feat_Y = self.LossNetwork(Y)

        loss = calculate_CX_Loss(feat_X.relu4_3, feat_Y.relu4_3)

        return loss

    def forward(self, output: "Tensor", target: "Tensor"):
        device = output.device

        self.total_loss = torch.tensor(0.0).to(device)
        self.adversarial_loss = torch.tensor(0.0).to(device)
        self.contextual_loss = torch.tensor(0.0).to(device)
        self.perception_loss = torch.tensor(0.0).to(device)
        self.image_loss = torch.tensor(0.0).to(device)
    

        if self.args.lambda_image:
            self.image_loss += (
                F.mse_loss(output, target).mean() * self.args.lambda_image
            )

        if self.args.lambda_l1:
            self.image_loss += (
                F.l1_loss(output, target).mean() * self.args.lambda_l1
            )

        if self.args.lambda_perception:
            self.perception_loss += (
                self._perception_metric(output, target).to(device)
                * self.args.lambda_perception
            )

        if self.args.lambda_contextual:
            self.contextual_loss += (
                self._contextual_metric(output, target).to(device)
                * self.args.lambda_contextual
            )


       
     
        self.total_loss += (
            self.image_loss
            + self.contextual_loss
            + self.perception_loss
        )
        # if self.is_multi:
        #     self.fft_loss = []
        #     self.fft_loss.append(self.image_loss.clone() + self.perception_loss.clone())
        #     if self.args.multi > 1:
        #         output_grid = split_tensor_into_grid(output, self.args.multi - 1)
        #         target_grid = split_tensor_into_grid(target, self.args.multi - 1)
        #         for i in range(0, self.args.multi - 1):
        #             if self.args.lambda_image:
        #                 image_loss = (
        #                     F.mse_loss(output_grid[i], target_grid[i]).mean() * self.args.lambda_image
        #                 ).to(device)

        #             if self.args.lambda_perception:
        #                 perception_loss = (
        #                     self._perception_metric(output_grid[i], target_grid[i]).to(device)
        #                     * self.args.lambda_perception
        #                 ).to(device)
        #             fft_loss = image_loss 
        #             fft_loss += perception_loss
        #             self.fft_loss.append(fft_loss)
                #split the original images 
                

        return self.total_loss


class DLoss(nn.Module):
    def __init__(self, args):
        super(DLoss, self).__init__()
        self.args = args

    def forward(self, real_logit, fake_logit):
        self.real_loss = F.binary_cross_entropy_with_logits(
            real_logit, soft_ones_like(real_logit)
        )
        self.fake_loss = F.binary_cross_entropy_with_logits(
            fake_logit, soft_zeros_like(fake_logit)
        )

        self.total_loss = (self.real_loss + self.fake_loss) / 2.0

        return self.total_loss


class Vgg16FeatureExtractor(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16FeatureExtractor, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        # h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        # vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        vgg_outputs = namedtuple("VggOutputs", ["relu2_2", "relu3_3", "relu4_3"])
        # out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        out = vgg_outputs(h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels=3, kernel_size=21, sigma=3, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
