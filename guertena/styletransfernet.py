"""Implementation of a style transfer network

References:
    - Gatys et al. - A Neural Algorithm of Artistic Style - https://arxiv.org/abs/1508.06576
    - Pytorch Neural Style tutorial: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    - Torch implementation by jcjohnson https://github.com/jcjohnson/neural-style
    - Pytorch implementation by ProGamerGov https://github.com/ProGamerGov/neural-style-pt
"""
from collections.abc import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



def get_device():
    """Returns the Pytorch device into which to run the calculations"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StyleTransferNet(torch.nn.Module):
    """Neural network that implements the style transfer algorithm of Gatys"""
    vgg19_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(get_device())
    vgg19_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(get_device())
    #content_layers_default = ['conv_4']
    content_layers_default = ['conv_10']  # Block 4, convolution 2
    #content_layers_default = ['relu_10']  # Block 4, convolution 2 (Gatys original)
    #style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    style_layers_default = ['conv_1', 'conv_3', 'conv_5', 'conv_9', 'conv_13']  # First convolution from each block
    #style_layers_default = ['relu_1', 'relu_3', 'relu_5', 'relu_9', 'relu_13']  # First convolution from each block (Gatys original)

    def __init__(self, content_img, style_img, content_weight, style_weight, tv_weight, content_layers=None, style_layers=None):
        super(StyleTransferNet, self).__init__()

        # Check inputs
        if content_layers is None:
            content_layers = list(self.content_layers_default)
        else:
            assert isinstance(content_layers, Iterable) and all([isinstance(x, str) for x in content_layers]), ValueError(f"content_layers must be a list of layer names, got {content_layers}")
        if style_layers is None:
            style_layers = list(self.style_layers_default)
        else:
            assert isinstance(style_layers, Iterable) and all([isinstance(x, str) for x in style_layers]), ValueError(f"style_layers must be a list of layer names, got {style_layers}")

        # Initialize VGG19 reference architecture
        cnn = models.vgg19(pretrained=True).features.to(get_device()).eval()

        # Start with image normalization layer
        normalization = Normalization(self.vgg19_normalization_mean, self.vgg19_normalization_std)
        self.model = nn.Sequential(normalization)

        # Add TV loss layer
        self.tv_loss = TVLoss(tv_weight)
        self.model.add_module("tv_loss", self.tv_loss)

        # Add VGG19 layers, with added losses
        self.content_losses = []
        self.style_losses = []
        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            self.model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = self.model(content_img).detach()
                content_loss = ContentLoss(target, content_weight)
                self.model.add_module("content_loss_{}".format(i), content_loss)
                self.content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = self.model(style_img).detach()
                style_loss = StyleLoss(target_feature, style_weight)
                self.model.add_module("style_loss_{}".format(i), style_loss)
                self.style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(self.model) - 1, -1, -1):
            if isinstance(self.model[i], ContentLoss) or isinstance(self.model[i], StyleLoss):
                break

        self.model = self.model[:(i + 1)]

    def forward(self, x):
        return self.model(x)


class ScaleGradients(torch.autograd.Function):
    """Scale gradients in the backward pass"""
    @staticmethod
    def forward(self, input_tensor, strength):
        self.strength = strength
        return input_tensor

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input / (torch.norm(grad_input, keepdim=True) + 1e-8)
        return grad_input * self.strength, None


class ContentLoss(nn.Module):
    """Style transfer content loss: penalizes differences between a target image and generated image activations"""
    def __init__(self, target, content_weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach()  # Do not compute gradients for target image
        self.content_weight = content_weight

    def forward(self, input):
        #self.loss = ScaleGradients.apply(F.mse_loss(input, self.target), self.content_weight) * self.content_weight
        # Note original Gatys paper loss is sum * 1/2 * weight, while this is mean * weight, but this implementation is compatible with jcjohnson
        self.loss = F.mse_loss(input, self.target) * self.content_weight
        return input


def gram_matrix(input):
    """Computes the Gram matrix of a single image after a convolution layer"""
    a, b, c, d = input.size()  # a=batch size(=1)
                               # b=number of feature maps (convolution channels)
                               # (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    # Note in Gatys paper this normalization is done in the loss, not here, but it's equivalent
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    """Style transfer style loss: penalizes differences between target image and generated image Gram matrices (activation correlations)"""
    def __init__(self, target, style_weight):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target).detach()
        self.style_weight = style_weight

    def forward(self, input):
        G = gram_matrix(input)
        #self.loss = ScaleGradients.apply(F.mse_loss(G, self.target), self.style_weight) * self.style_weight
        # Note original Gatys paper loss is sum * 1/4 * weight, while this is mean * weight, but this implementation is compatible with jcjohnson
        self.loss = F.mse_loss(G, self.target) * self.style_weight
        return input


class TVLoss(nn.Module):
    """Total Variation loss: penalizes neighbouring with different values"""

    def __init__(self, strength):
        super(TVLoss, self).__init__()
        self.strength = strength

    def forward(self, input):
        self.x_diff = input[:,:,1:,:] - input[:,:,:-1,:]
        self.y_diff = input[:,:,:,1:] - input[:,:,:,:-1]
        self.loss = self.strength / torch.numel(input) * (torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff)))
        return input


class Normalization(nn.Module):
    """Layer that normalizes an image using given mean and standard deviation statistics"""
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
