import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import get_window

class Invertible1x1Conv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Initialize with a random orthogonal matrix
        W = torch.qr(torch.FloatTensor(c, c).normal_()[0])
        # Ensure determinant is 1.0
        if torch.det(W) < 0:
            W[:,0] = -1 * W[:,0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z, reverse=False):
        batch_size, group_size, n_of_groups = z.size()
        W = self.conv.weight.squeeze()
        
        if reverse:
            if not hasattr(self, 'W_inverse'):
                W_inverse = W.float().inverse()
                W_inverse = W_inverse[..., None]
                self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            log_det_W = batch_size * n_of_groups * torch.logdet(W)
            z = self.conv(z)
            return z, log_det_W

class WN(nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(hidden_channels % 2 == 0)
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                 dilation=dilation, padding=padding)
            in_layer = nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)
            
            # Last layer doesn't need res_skip layer
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
                
            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])
        
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            
            acts = fused_add_tanh_sigmoid_multiply(
                x_in, n_channels_tensor)
            
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:,:self.hidden_channels,:]
                x = (x + res_acts)
                output = output + res_skip_acts[:,self.hidden_channels:,:]
            else:
                output = output + res_skip_acts
        
        return output

def fused_add_tanh_sigmoid_multiply(input_a, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts

class WaveGlow(nn.Module):
    def __init__(self, n_mel_channels, n_flows, n_group, n_early_every, 
                 n_early_size, WN_config):
        super().__init__()
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.WN_config = WN_config
        
        self.upsample = nn.ConvTranspose1d(
            n_mel_channels, n_mel_channels, 1024, stride=256)
        
        assert(n_group % 2 == 0)
        self.n_remaining_channels = n_group
        
        self.convinv = nn.ModuleList()
        self.WN = nn.ModuleList()
        
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                self.n_remaining_channels -= n_early_size
                
            self.convinv.append(Invertible1x1Conv(self.n_remaining_channels))
            
            self.WN.append(WN(
                self.n_remaining_channels, 
                WN_config['kernel_size'], 
                WN_config['dilation_rate'], 
                WN_config['n_layers']))
        
        self.n_remaining_channels -= n_early_size

    def forward(self, forward_input):
        audio, spect = forward_input
        
        spect = self.upsample(spect)
        assert(spect.size(2) >= audio.size(1))
        
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, :audio.size(1)]
        
        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1)
        spect = spect.permute(0, 2, 1)
        
        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
        output_audio = []
        log_s_list = []
        log_det_W_list = []
        
        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:,:self.n_early_size,:])
                audio = audio[:,self.n_early_size:,:]
            
            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)
            
            n_half = int(audio.size(1)/2)
            audio_0 = audio[:,:n_half,:]
            audio_1 = audio[:,n_half:,:]
            
            output = self.WN[k]((audio_0, spect))
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(log_s)*audio_1 + b
            log_s_list.append(log_s)
            
            audio = torch.cat([audio_0, audio_1], 1)
        
        output_audio.append(audio)
        return torch.cat(output_audio, 1), log_s_list, log_det_W_list

    def infer(self, spect, sigma=1.0):
        spect = self.upsample(spect)
        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff]
        
        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1)
        spect = spect.permute(0, 2, 1)
        
        audio = sigma * torch.randn(
            spect.size(0), self.n_remaining_channels, spect.size(2), 
            device=spect.device).to(spect.dtype)
        
        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1)/2)
            audio_0 = audio[:,:n_half,:]
            audio_1 = audio[:,n_half:,:]
            
            output = self.WN[k]((audio_0, spect))
            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)
            
            audio = self.convinv[k](audio, reverse=True)
            
            if k % self.n_early_every == 0 and k > 0:
                z = sigma * torch.randn(
                    spect.size(0), self.n_early_size, spect.size(2), 
                    device=spect.device).to(spect.dtype)
                audio = torch.cat((z, audio), 1)
        
        return audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1)
