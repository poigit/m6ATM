# basic
import os, math
import numpy as np

# pytorch
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.autograd import Variable, Function

class WaveNetModel(nn.Module):
    
    '''
    A Complete Wavenet Model

    Args:
        layers (Int):               Number of layers in each block
        blocks (Int):               Number of wavenet blocks of this model
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        input_channels (Int):       Number of channels of input tensor
        kernel_size (Int):          Size of the dilation kernel
        dtype:                      Parameter type of this model

    Shape:
        - Input: (n, channel, length)
    '''
    
    def __init__(self,
                 layers = 3,
                 blocks = 2,
                 dilation_channels = 32,
                 residual_channels = 32,
                 skip_channels = 256,
                 end_channels = 128,
                 input_channels = 5,
                 last_channels = 10144,
                 kernel_size = 2,
                 num_classes = 2,
                 dropout = 0.2,
                 dtype = torch.FloatTensor,
                 bias = False):

        super(WaveNetModel, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.input_channels = input_channels
        self.last_channels = last_channels
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.num_classes = num_classes

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []
        self.dilated_queues = []
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels = self.input_channels,
                                    out_channels = residual_channels,
                                    kernel_size = 1,
                                    bias = bias)

        for n, b in enumerate(range(blocks)):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))

                # dilated queues for fast generation
                self.dilated_queues.append(DilatedQueue(max_length = (kernel_size-1)*new_dilation+1,
                                                        num_channels = residual_channels,
                                                        dilation = new_dilation,
                                                        dtype = dtype))

                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels = residual_channels,
                                                   out_channels = dilation_channels,
                                                   kernel_size = kernel_size,
                                                   bias = bias))

                self.gate_convs.append(nn.Conv1d(in_channels = residual_channels,
                                                 out_channels = dilation_channels,
                                                 kernel_size = kernel_size,
                                                 bias = bias))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels = dilation_channels,
                                                     out_channels = residual_channels,
                                                     kernel_size = 1,
                                                     bias = bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels = dilation_channels,
                                                 out_channels = skip_channels,
                                                 kernel_size = 1,
                                                 bias = bias))

                receptive_field+=additional_scope
                additional_scope*=2
                init_dilation = new_dilation
                new_dilation*=2

        self.end_conv_1 = nn.Conv1d(in_channels = skip_channels,
                                    out_channels = end_channels,
                                    kernel_size = 1,
                                    bias = True)

        self.end_conv_2 = nn.Conv1d(in_channels = end_channels,
                                    out_channels = int(end_channels/4),
                                    kernel_size = 1,
                                    bias = True)
        
        self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 2)
        self.dropout = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.receptive_field = receptive_field
        self.relu = nn.ReLU()
        
        ### Dense layer
        self.output_layer = nn.Sequential(nn.Linear(self.last_channels, 1024), # maxpool out_size
                                          nn.ReLU(),
                                          nn.Dropout(dropout),
                                          nn.Linear(1024, 256),
                                          nn.ReLU(),
                                          nn.Dropout(dropout),
                                          nn.Linear(256, num_classes))
        
        
        self.fc1 = nn.Sequential(nn.Linear(self.last_channels, 1024)) # maxpool out_size # 32*317
        self.fc2 = nn.Sequential(nn.Linear(1024, 256),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(256, num_classes)) # maxpool out_size
        
        
    def wavenet(self, input, dilation_func):

        x = self.start_conv(input) # [1, 1280] -> [32, 256]
        skip = 0

        # WaveNet layers
        for i in range(self.blocks*self.layers):
            
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            (dilation, init_dilation) = self.dilations[i]
            residual = dilation_func(x, dilation, init_dilation, i)
      
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate 

            # parametrized skip connection
            s = x
            if x.size(2)!=1:
                 s = dilate(x, 1, init_dilation = dilation)
            s = self.skip_convs[i](s)
            
            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
                
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, (self.kernel_size - 1):]

        x = F.relu(skip)

        return x

    def wavenet_dilate(self, input, dilation, init_dilation, i):
        
        x = dilate(input, dilation, init_dilation)
        
        return x

    def queue_dilate(self, input, dilation, init_dilation, i):
        
        queue = self.dilated_queues[i]
        queue.enqueue(input.data[0])
        x = queue.dequeue(num_deq = self.kernel_size,
                          dilation = dilation)
        x = x.unsqueeze(0)

        return x

    def forward(self, input):
        
        x = self.wavenet(input, dilation_func = self.wavenet_dilate)
        x = F.relu(self.end_conv_1(x))
        x = self.maxpool(x)
        x = F.relu(self.end_conv_2(x))
        x = self.maxpool(x) 
        
        x = x.view(x.size(0), -1)
        x = self.output_layer(x)
        
        Y_hat = torch.argmax(x, dim = 1)
        Y_prob = F.softmax(x, dim = 1).to('cpu').detach().numpy().copy()
        
        results_dict = {'value': x, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        
        return results_dict
    
    
    def encoder(self, input):
        
        x = self.wavenet(input, dilation_func = self.wavenet_dilate)
        x = F.relu(self.end_conv_1(x))
        x = self.maxpool(x)
        x = F.relu(self.end_conv_2(x))
        x = self.maxpool(x)
        
        x = x.view(x.size(0), -1)
        feats = self.fc1(x)
        y = self.fc2(self.dropout(self.relu(feats)))
        
        return feats, y
    
    


def dilate(x, dilation, init_dilation = 1, pad_start = True):
    
    '''
    :param x: Tensor of size (N, C, L), where N is the input dilation, C is the number of channels, and L is the input length
    :param dilation: Target dilation. Will be the size of the first dimension of the output tensor.
    :param pad_start: If the input length is not compatible with the specified dilation, zero padding is used. This parameter determines 
                      whether the zeros are added at the start or at the end.
    :return: The dilated tensor of size (dilation, C, L*N / dilation). The output might be zero padded at the start
    '''

    [n, c, l] = x.size()
    dilation_factor = dilation/init_dilation
    if dilation_factor == 1:
        return x

    # zero padding for reshaping
    new_l = int(np.ceil(l/dilation_factor)*dilation_factor)
    if new_l!=l:
        l = new_l
        x = constant_pad_1d(x, new_l, dimension = 2, pad_start = pad_start)

    l_old = int(round(l/dilation_factor))
    n_old = int(round(n*dilation_factor))
    l = math.ceil(l * init_dilation/dilation)
    n = math.ceil(n * dilation/init_dilation)

    # reshape according to dilation
    x = x.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
    x = x.view(c, l, n)
    x = x.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

    return x  

class DilatedQueue:
    
    def __init__(self, max_length, data = None, dilation = 1, num_deq = 1, num_channels = 1, dtype = torch.FloatTensor):
        self.in_pos = 0
        self.out_pos = 0
        self.num_deq = num_deq
        self.num_channels = num_channels
        self.dilation = dilation
        self.max_length = max_length
        self.data = data
        self.dtype = dtype
        if data == None:
            self.data = Variable(dtype(num_channels, max_length).zero_())

    def enqueue(self, input):
        
        self.data[:, self.in_pos] = input
        self.in_pos = (self.in_pos + 1) % self.max_length

    def dequeue(self, num_deq = 1, dilation = 1):
        
        #       |
        #  |6|7|8|1|2|3|4|5|
        #         |
        start = self.out_pos - ((num_deq - 1) * dilation)
        if start < 0:
            t1 = self.data[:, start::dilation]
            t2 = self.data[:, self.out_pos % dilation:self.out_pos + 1:dilation]
            t = torch.cat((t1, t2), 1)
        else:
            t = self.data[:, start:self.out_pos + 1:dilation]

        self.out_pos = (self.out_pos + 1) % self.max_length
        return t

    def reset(self):
        self.data = Variable(self.dtype(self.num_channels, self.max_length).zero_())
        self.in_pos = 0
        self.out_pos = 0
        
def constant_pad_1d(input, target_size, dimension = 0, value = 0, pad_start = False):
    
    pads = [0]*(input.ndim*2)
    pads[2*dimension+(1 if pad_start else 0)] = target_size-input.shape[dimension]
    
    return torch.nn.functional.pad(input, pads[::-1], mode = 'constant', value = value)