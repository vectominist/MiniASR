'''
    File      [ model_utils.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Some utilities for models. ]
'''


def freeze_model(model):
    ''' Freeze all parameters in a model. '''
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    ''' Un-freeze all parameters in a model. '''
    for param in model.parameters():
        param.requires_grad = True


def check_extractor_attributes(model, device='cuda:0'):
    ''' Checks extractor's attributes. '''

    import torch
    print(f'Testing {model} on {device}')

    extractor = torch.hub.load('s3prl/s3prl', model).to(device)
    extractor.eval()
    wavs = [torch.zeros(160000, dtype=torch.float).to(device)
            for _ in range(8)]
    with torch.no_grad():
        hiddens = extractor(wavs)
        print(f'Hidden feature types: {hiddens.keys()}')

    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)

    print('Used {:.2f} MB of GPU VRAM with 8 audio samples each having 10 secs.'
          .format(info.used / 1024 / 1024))


if __name__ == '__main__':
    import sys
    check_extractor_attributes(sys.argv[1])
