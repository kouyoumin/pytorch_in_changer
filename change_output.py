import torch.nn as nn


def change_classifier_out_features(model, new_out_features=1):
    for n, m in model.named_children():
        if n == 'fc' or n == 'classifier':
            print('Found classifier (%s): %s' % (n, m))
            if isinstance(m, nn.modules.conv._ConvNd):
                cls_class = m.__class__
                setattr(model, n, cls_class(m.in_channels, new_out_features, m.kernel_size, m.stride, m.padding, m.dilation, m.groups, False if m.bias==None else True, m.padding_mode))
            elif isinstance(m, nn.Linear):
                cls_class = m.__class__
                setattr(model, n, cls_class(m.in_features, new_out_features, False if m.bias==None else True))
            elif isinstance(m, nn.Sequential):
                for idx in reversed(range(len(m))):
                    if isinstance(m[idx], nn.modules.conv._ConvNd):
                        cls_class = m[idx].__class__
                        m[idx] = cls_class(m[idx].in_channels, new_out_features, m[idx].kernel_size, m[idx].stride, m[idx].padding, m[idx].dilation, m[idx].groups, False if m[idx].bias==None else True, m[idx].padding_mode)
                    elif isinstance(m[idx], nn.Linear):
                        cls_class = m[idx].__class__
                        m[idx] = cls_class(m[idx].in_features, new_out_features, False if m[idx].bias==None else True)

            break
    
    for n, m in model.named_children():
        if n == 'fc' or n == 'classifier':
            print('Changed classifier (%s) to: %s' % (n, m))
            break


def test(model_str, new_out_features):
    import torch
    import torchvision
    model = torchvision.models.__dict__[model_str](pretrained=True)
    model.eval()
    #dummy_data = torch.randn(1, 1, 32, 32)
    for n,m in model.named_children():
        if n == 'fc' or n == 'classifier':
            sd = m.state_dict()
            break
    
    change_classifier_out_features(model, new_out_features)
    
    for n,m in model.named_modules():
        if n == 'fc' or n == 'classifier':
            new_param = list(m.parameters())
            assert(new_param[-1].shape[0] == new_out_features)
            break


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print('Usage: %s [model] [out_features]' % (sys.argv[0]))
    test(sys.argv[1], int(sys.argv[2]))
