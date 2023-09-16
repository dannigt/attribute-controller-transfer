import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules.fairseq_dropout import FairseqDropout

# From https://github.com/cbaziotis/fairseq/blob/hyperadapters/fairseq/modules/adapter.py#L30-L35
def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class ClassificationLayer(nn.Module):
    def __init__(self, args, input_dim, output_dim, middle_dim=512):
        super(ClassificationLayer, self).__init__()
        self.fc_1 = Linear(input_dim, middle_dim)
        self.fc_2 = Linear(middle_dim, output_dim)
        self.dropout = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )

    def forward(self, x):
        x = F.relu(self.fc_1(x), inplace=True)
        x = self.dropout(x)
        x = self.fc_2(x) #, inplace=True)
        return x
