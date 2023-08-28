import torch.nn as nn


#### CNN Model


class CNNModel(nn.Module):
    def __init__(self, in_dim, num_classes, dropout=0.0):
        super(CNNModel, self).__init__()

        self.conv_layer1 = self._conv_layer_set1(in_dim, 32)
        self.conv_layer2 = self._conv_layer_set2(32, 64)
        self.fc = nn.Linear(6400, num_classes)
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        #self.batch = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=dropout)
    
    def _conv_layer_set1(self, _in, _out):
        conv_layer = nn.Sequential(
            nn.Conv3d(_in, _out, kernel_size=(1,11,11), stride=1, padding=0),
            nn.MaxPool3d((15,2,2))
        )
        return conv_layer
    
    def _conv_layer_set2(self, _in, _out):
        conv_layer = nn.Sequential(
            nn.Conv3d(_in, _out, kernel_size=(1,5,5), stride=1, padding=0),
            nn.MaxPool3d((5,5,5))
        )
        return conv_layer
    
    def forward(self, x):

        ## 2 steps convolution
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        
        ## flatten
        out = out.view(out.size(0), -1)

        ## MLP
        out = self.fc(out)
        return out

