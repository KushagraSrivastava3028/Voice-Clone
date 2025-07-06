import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .text.symbols import symbols

class Encoder(nn.Module):
    """Tacotron2 Encoder"""
    def __init__(self, embed_dims, num_chars, encoder_dims, dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, embed_dims)
        padding_idx = 0
        self.embedding.weight.data[0].fill_(0)
        
        self.convolutions = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dims, encoder_dims, 5, padding=2),
                nn.BatchNorm1d(encoder_dims),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Conv1d(encoder_dims, encoder_dims, 5, padding=2),
                nn.BatchNorm1d(encoder_dims),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Conv1d(encoder_dims, encoder_dims, 5, padding=2),
                nn.BatchNorm1d(encoder_dims),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        ])
        
        self.lstm = nn.LSTM(encoder_dims, encoder_dims // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        x = self.embedding(x).transpose(1, 2)
        
        for conv in self.convolutions:
            x = conv(x)
        
        x = x.transpose(1, 2)
        
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        
        return outputs

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super().__init__()
        padding = (attention_kernel_size - 1) // 2
        self.location_conv = nn.Conv2d(
            2, attention_n_filters,
            kernel_size=(attention_kernel_size, 1),
            padding=(padding, 0),
            bias=False)
        self.location_dense = nn.Linear(
            attention_n_filters, attention_dim, bias=False)

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention

class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim, 
                 attention_location_n_filters, attention_location_kernel_size):
        super().__init__()
        self.query_layer = nn.Linear(attention_rnn_dim, attention_dim, bias=False)
        self.memory_layer = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + self.memory_layer(processed_memory)))
        
        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory, 
                attention_weights_cat, mask):
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)
        
        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)
            
        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        
        return attention_context, attention_weights

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes, dropout=0.5):
        super().__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for (in_size, out_size) in zip(in_sizes, sizes)
        ])
        self.dropout = dropout

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=self.dropout, training=True)
        return x

class Postnet(nn.Module):
    """Tacotron2 Postnet"""
    def __init__(self, n_mel_channels, postnet_embedding_dims, postnet_kernel_size, postnet_n_convolutions, dropout=0.5):
        super().__init__()
        self.convolutions = nn.ModuleList()
        
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(n_mel_channels, postnet_embedding_dims,
                         kernel_size=postnet_kernel_size, padding=int((postnet_kernel_size-1)/2)),
                nn.BatchNorm1d(postnet_embedding_dims),
                nn.Tanh(),
                nn.Dropout(dropout)
            )
        )
        
        for _ in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(postnet_embedding_dims, postnet_embedding_dims,
                             kernel_size=postnet_kernel_size, padding=int((postnet_kernel_size-1)/2)),
                    nn.BatchNorm1d(postnet_embedding_dims),
                    nn.Tanh(),
                    nn.Dropout(dropout)
                )
            )
        
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(postnet_embedding_dims, n_mel_channels,
                         kernel_size=postnet_kernel_size, padding=int((postnet_kernel_size-1)/2)),
                nn.BatchNorm1d(n_mel_channels),
                nn.Dropout(dropout)
            )
        )

    def forward(self, x):
        for conv in self.convolutions:
            x = conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self, n_mel_channels, n_frames_per_step, 
                 encoder_embedding_dim, attention_rnn_dim, decoder_rnn_dim,
                 prenet_dim, max_decoder_steps, gate_threshold,
                 p_attention_dropout, p_decoder_dropout,
                 attention_dim, attention_location_n_filters,
                 attention_location_kernel_size):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        
        self.prenet = Prenet(
            n_mel_channels * n_frames_per_step,
            [prenet_dim, prenet_dim],
            p_decoder_dropout)
        
        self.attention_rnn = nn.LSTMCell(
            prenet_dim + encoder_embedding_dim,
            attention_rnn_dim)
        
        self.attention_layer = Attention(
            attention_rnn_dim, encoder_embedding_dim,
            attention_dim, attention_location_n_filters,
            attention_location_kernel_size)
        
        self.decoder_rnn = nn.LSTMCell(
            attention_rnn_dim + encoder_embedding_dim,
            decoder_rnn_dim, 1)
        
        self.linear_projection = nn.Linear(
            decoder_rnn_dim + encoder_embedding_dim,
            n_mel_channels * n_frames_per_step)
        
        self.gate_layer = nn.Linear(
            decoder_rnn_dim + encoder_embedding_dim, 1,
            bias=True)
        
    def get_go_frame(self, memory):
        B = memory.size(0)
        return torch.zeros(
            B, self.n_mel_channels * self.n_frames_per_step,
            device=memory.device)

    def initialize_decoder_states(self, memory, mask):
        B, T, _ = memory.size()
        
        self.attention_hidden = torch.zeros(
            B, self.attention_rnn_dim, device=memory.device)
        self.attention_cell = torch.zeros(
            B, self.attention_rnn_dim, device=memory.device)
        
        self.decoder_hidden = torch.zeros(
            B, self.decoder_rnn_dim, device=memory.device)
        self.decoder_cell = torch.zeros(
            B, self.decoder_rnn_dim, device=memory.device)
        
        self.attention_weights = torch.zeros(
            B, T, device=memory.device)
        self.attention_weights_cum = torch.zeros(
            B, T, device=memory.device)
        self.attention_context = torch.zeros(
            B, self.encoder_embedding_dim, device=memory.device)
        
        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        return decoder_inputs.reshape(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1)
        mel_outputs = mel_outputs.reshape(
            mel_outputs.size(0), -1, self.n_mel_channels)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        alignments = torch.stack(alignments).transpose(0, 1)
        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, 0.1, self.training)
        
        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)
        
        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, 0.1, self.training)
        
        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)
        
        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)
        
        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))
        
        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(decoder_input)
            
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]
        
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        
        return mel_outputs, gate_outputs, alignments

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.num_mels = hparams['n_mel_channels']
        self.n_frames_per_step = hparams['n_frames_per_step']
        
        self.embedding = nn.Embedding(
            len(symbols), hparams['symbols_embedding_dim'])
        padding_idx = 0
        self.embedding.weight.data[0].fill_(0)
        
        self.encoder = Encoder(
            hparams['encoder_embedding_dim'],
            len(symbols),
            hparams['encoder_dims'],
            hparams['dropout'])
        
        self.decoder = Decoder(
            hparams['n_mel_channels'],
            hparams['n_frames_per_step'],
            hparams['encoder_dims'],
            hparams['attention_rnn_dim'],
            hparams['decoder_rnn_dim'],
            hparams['prenet_dim'],
            hparams['max_decoder_steps'],
            hparams['gate_threshold'],
            hparams['p_attention_dropout'],
            hparams['p_decoder_dropout'],
            hparams['attention_dim'],
            hparams['attention_location_n_filters'],
            hparams['attention_location_kernel_size'])
        
        self.postnet = Postnet(
            hparams['n_mel_channels'],
            hparams['postnet_embedding_dims'],
            hparams['postnet_kernel_size'],
            hparams['postnet_n_convolutions'],
            hparams['dropout'])

    def forward(self, text, text_lengths, mels, output_lengths):
        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)
        
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        
        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)
        
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        
        return mel_outputs_postnet, gate_outputs, alignments
