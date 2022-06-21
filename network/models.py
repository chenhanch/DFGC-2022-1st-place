import torch
import torch.nn as nn
from torch.nn import init
import torchvision
from efficientnet_pytorch import EfficientNet
import timm
from pytorch_pretrained_vit import ViT


# fc layer weight init
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


# 当in_channel != 3 时，初始化模型的第一个Conv的weight， 把之前的通道copy input_chaneel/3 次
def init_imagenet_weight(_conv_stem_weight, input_channel=3):
    for i in range(input_channel//3):
        if i == 0:
            _conv_stem_weight_new = _conv_stem_weight
        else:
            _conv_stem_weight_new = torch.cat([_conv_stem_weight_new, _conv_stem_weight], axis=1)

    return torch.nn.Parameter(_conv_stem_weight_new)


class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """
    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        if modelchoice == 'efficientnet-b7' or modelchoice == 'efficientnet-b6'\
                or modelchoice == 'efficientnet-b5' or modelchoice == 'efficientnet-b4'\
                or modelchoice == 'efficientnet-b3' or modelchoice == 'efficientnet-b2'\
                or modelchoice == 'efficientnet-b1' or modelchoice == 'efficientnet-b0':
            # self.model = EfficientNet.from_name(modelchoice, override_params={'num_classes': num_out_classes})
            self.model = get_efficientnet(model_name=modelchoice, num_classes=num_out_classes)
        elif modelchoice == 'tf_efficientnet_b7_ns' or modelchoice == 'tf_efficientnet_b6_ns' \
                or modelchoice == 'tf_efficientnet_b5_ns' or modelchoice == 'tf_efficientnet_b4_ns' \
                or modelchoice == 'tf_efficientnet_b3_ns' or modelchoice == 'tf_efficientnet_b2_ns':
            self.model = get_efficientnet_ns(model_name=modelchoice, pretrained=True,  num_classes=2)
        elif modelchoice == 'resnet18' or modelchoice == 'resnet50' or modelchoice == 'resnet101' or modelchoice == 'resnet152':
            if modelchoice == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=True)
            elif modelchoice == 'resnet18':
                self.model = torchvision.models.resnet18(pretrained=True)
            elif modelchoice == 'resnet101':
                self.model = torchvision.models.resnet101(pretrained=True)
            elif modelchoice == 'resnet152':
                self.model = torchvision.models.resnet152(pretrained=True)
            # Replace fc
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
                init.normal_(self.model.fc.weight.data, std=0.001)
                init.constant_(self.model.fc.bias.data, 0.0)
            else:
                self.model.fc = nn.Sequential(
                    nn.Linear(num_ftrs, 256),
                    nn.Dropout(p=dropout),
                    nn.Linear(256, num_out_classes)
                )
                init.normal_(self.model.fc[2].weight.data, std=0.001)
                init.constant_(self.model.fc[2].bias.data, 0.0)
        elif modelchoice == 'B_16_imagenet1k':
            self.model = ViT(modelchoice, pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_out_classes)
            init.normal_(self.model.fc.weight.data, std=0.001)
            init.constant_(self.model.fc.bias.data, 0.0)
        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def forward(self, x):

        x = self.model(x)
        return x


def model_selection(modelname, num_out_classes, dropout=None):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """
    if modelname == 'resnet18' or modelname == 'resnet50' or modelname == 'resnet101' or modelname == 'resnet152':
        return TransferModel(modelchoice=modelname, dropout=dropout,
                             num_out_classes=num_out_classes), \
               224, True, ['image'], None
    elif modelname == 'efficientnet-b7' or modelname == 'efficientnet-b6'\
            or modelname == 'efficientnet-b5' or modelname == 'efficientnet-b4' \
            or modelname == 'efficientnet-b3' or modelname == 'efficientnet-b2' \
            or modelname == 'efficientnet-b1' or modelname == 'efficientnet-b0':
        return TransferModel(modelchoice=modelname, dropout=dropout,
                             num_out_classes=num_out_classes), \
               224, True, ['image'], None
    elif modelname == 'tf_efficientnet_b7_ns' or modelname == 'tf_efficientnet_b6_ns'\
            or modelname == 'tf_efficientnet_b5_ns' or modelname == 'tf_efficientnet_b4_ns' \
            or modelname == 'tf_efficientnet_b3_ns' or modelname == 'tf_efficientnet_b2_ns' \
            or modelname == 'tf_efficientnet_b1_ns' or modelname == 'tf_efficientnet_b0_ns':
        return TransferModel(modelchoice=modelname, dropout=dropout,
                             num_out_classes=num_out_classes), \
               224, True, ['image'], None
    elif modelname == 'B_16_imagenet1k':
        return TransferModel(modelchoice=modelname, dropout=dropout,
                             num_out_classes=num_out_classes), \
               384, True, ['image'], None
    else:
        raise NotImplementedError(modelname)


def get_efficientnet(model_name='efficientnet-b0', num_classes=2, start_down=True):
    net = EfficientNet.from_pretrained(model_name)
    if not start_down:
        net._conv_stem.stride = 1
    # net = EfficientNet.from_name(model_name)
    in_features = net._fc.in_features
    net._fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    return net


def get_efficientnet_ns(model_name='tf_efficientnet_b3_ns', pretrained=True, num_classes=2, start_down=True):
    """
     # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    :param model_name:
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    if not start_down:
        net.conv_stem.stride = (1, 1)
    n_features = net.classifier.in_features
    net.classifier = nn.Linear(n_features, num_classes)

    return net


def get_vit(model_name='B_16_imagenet1k', num_classes=2, pretrained=True):
    model = ViT(model_name, pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def get_swin_transformers(model_name='swin_base_patch4_window12_384', pretrained=True, num_classes=2):
    """
    :param model_name: swin_base_patch4_window12_384   swin_large_patch4_window12_384
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.head.in_features
    net.head = nn.Linear(n_features, num_classes)

    return net

def get_convnext(model_name='convnext_xlarge_384_in22ft1k', pretrained=True, num_classes=2):
    """
    :param model_name: convnext_xlarge_384_in22ft1k
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.head.fc.in_features
    net.head.fc = nn.Linear(n_features, num_classes)

    return net


def get_resnet200d(model_name='resnet200d', pretrained=True, num_classes=2):
    """
    :param model_name: resnet200d, input_size=512
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.fc.in_features
    net.fc = nn.Linear(n_features, num_classes)

    return net


if __name__ == '__main__':
    # model, image_size, *_ = model_selection('B_16_imagenet1k', num_out_classes=2)
    # model, image_size = get_vit(model_name='B_16_imagenet1k', num_classes=2, pretrained=True), 384
    # model, image_size = get_efficientnet(model_name='efficientnet-b0', num_classes=2, start_down=False), 512
    model, image_size = get_efficientnet_ns(model_name='tf_efficientnet_b0_ns', num_classes=2, start_down=False), 512
    # print(model.model.image_size)
    print(model)

    model = model.to(torch.device('cpu'))
    from torchsummary import summary
    input_s = (3, image_size, image_size)
    print(summary(model, input_s, device='cpu'))

    # print(model._modules.items())
    # print(model)

    pass

