'''
    File      [ feat_selection.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Feature selection layer. ]
'''

import torch
from torch import nn
from torch.nn import functional as F


class FeatureSelection(nn.Module):
    '''
        Feature selection for pre-trained audio extractors.
        extractor [nn.Module]: Pre-trained extractor
        selection [str]: selection method
    '''

    def __init__(self, extractor, selection='hidden_states'):
        super(FeatureSelection, self).__init__()

        self.selection = selection

        # Try a single sample
        wavs = [torch.zeros(160000, dtype=torch.float)]
        extractor.eval()
        with torch.no_grad():
            emb_dict = extractor(wavs)
        # Note: The extracted features should be in a dict
        self.n_features = len(emb_dict[selection])
        self.feat_dim = emb_dict[selection][0].shape[-1]

        if self.n_features == 1:
            # Only one type of feature
            self.weight_layer = None
        else:
            # Several types of features: perform weighted sum
            # This happends when using models like hubert or wav2vec2
            # since they have multiple hidden layers.
            self.weight_layer = nn.Parameter(
                torch.zeros(self.n_features, requires_grad=True))

    def forward(self, emb_dict):
        '''
            Feature selection process.
            Input:
                emb_dict [dict]: output of a pre-trained extractor
        '''

        feat = emb_dict[self.selection]

        if self.weight_layer is None:
            return F.layer_norm(feat[0], (feat[0].shape[-1], ))
        else:
            # Perform weighted sum
            feat = torch.stack(feat, dim=0)  # n_features x Batch x Time x Dim
            feat = F.layer_norm(feat, (feat.shape[-1], ))
            weights = torch.softmax(self.weight_layer, dim=0)
            feat = (feat * weights.reshape(self.n_features, 1, 1, 1)).sum(0)
            return feat
