import torch.nn as nn


def change_first_conv_in_channel(model, new_in_channels=1):
    for n, m in model.named_modules():
        if isinstance(m, nn.modules.conv._ConvNd):
            print('Found first conv layer (%s) to: %s' % (n, m))
            if m.in_channels != new_in_channels:
                # Get objects hierarchically
                hiers = n.split('.')
                levels = [model]
                for hier in hiers[:-1]:
                    levels.append(getattr(levels[-1], hier))
                # Assign new 
                conv_class = m.__class__
                setattr(levels[-1], hiers[-1], conv_class(new_in_channels, m.out_channels, m.kernel_size, m.stride, m.padding, m.dilation, m.groups, m.bias, m.padding_mode))
                new_sd = change_state_dict_in_channel(m.state_dict(), new_in_channels)
                getattr(levels[-1], hiers[-1]).load_state_dict(new_sd)
            break
    
    for n, m in model.named_modules():
        if isinstance(m, nn.modules.conv._ConvNd):
            print('Changed first conv layer (%s) to: %s' % (n, m))
            break


def change_state_dict_in_channel(state_dict, new_in_channels):
    new_state_dict = {}
    for key in state_dict:
        if key == 'weight':
            new_state_dict[key] = state_dict['weight'].sum(dim=1, keepdim=True).repeat(1, new_in_channels, 1, 1) / new_in_channels
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def test(model_str, new_in_channel):
    import torch
    import torchvision
    model = torchvision.models.__dict__[model_str](pretrained=True)
    model.eval()
    dummy_data = torch.randn(1,1,32,32)
    for n,m in model.named_modules():
        if isinstance(m, nn.modules.conv._ConvNd):
            sd = m.state_dict()
            with torch.no_grad():
                orig_conv_out = m(dummy_data.repeat(1,3,1,1))
                orig_model_out = model(dummy_data.repeat(1,3,1,1))
            break
    
    change_first_conv_in_channel(model, new_in_channel)
    
    for n,m in model.named_modules():
        if isinstance(m, nn.modules.conv._ConvNd):
            new_sd = m.state_dict()
            with torch.no_grad():
                new_conv_out = m(dummy_data)
                new_model_out = model(dummy_data)
            assert(new_sd['weight'].shape[1] == new_in_channel)
            assert(torch.equal(new_sd['weight'][:, :1, :, :], sd['weight'].sum(dim=1, keepdim=True)))
            assert(torch.allclose(orig_conv_out, new_conv_out, atol=1e-05))
            assert(torch.allclose(orig_model_out, new_model_out, atol=1e-05))
            break


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print('Usage: %s [model] [in_channels]' % (sys.argv[0]))
    test(sys.argv[1], int(sys.argv[2]))
