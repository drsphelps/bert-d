import torch
from transformers import BertModel
import numpy as np


def zeroing(linear, zero_type, share):
    head_offsets = range(0,768, 64)
    batch = 64
    if zero_type == 'random':
            np.random.seed(42)
            torch.manual_seed(42)
            # Serguei's approach to reduce running time
            for head in head_offsets:
                # update to unique random integers
                rnd_index = np.random.choice(range(head, head+64), int(batch*share), replace=False)
                for row in range(0, linear.weight.size()[0]):
                    linear.weight[row][rnd_index] = linear.weight[row][rnd_index].mul(0.0)
                    linear.bias[rnd_index] = linear.bias[rnd_index].mul(0.0)
            return model
    elif zero_type == 'first':
        offset = int(batch*share)
        for head in head_offsets:
            for row in range(linear.weight.size()[0]):
                linear.weight[row][head:head+offset] = linear.weight[row][head:head+offset].mul(0.0)
                linear.bias[head:head+offset] = linear.bias[head:head+offset].mul(0.0)
        return model
    elif zero_type == 'shuffle':
        offset = int(64*share)
        for head in head_offsets:
            for row in range(linear.weight.size()[0]):
                np.random.shuffle(linear.weight[row][head:head+offset])
                np.random.shuffle(linear.bias[row][head:head+offset])
        return model
    else:
        raise ValueError("zeroing type is not supported!")


def break_attn_heads_by_layer(model, tensors, zero_type='random', share=0.25, layer=0):
    # zeroing both weights and bias
    damaged_model = model
    with torch.no_grad():
        if 'value' in tensors:
            damaged_model = zeroing(damaged_model.encoder.layer[layer].attention.self.value, zero_type, share)
        if 'key' in tensors:
            damaged_model = zeroing(damaged_model.encoder.layer[layer].attention.self.key, zero_type, share)
        if 'query' in tensors:
            damaged_model = zeroing(damaged_model.encoder.layer[layer].attention.self.query, zero_type, share)
    return damaged_model


def break_attn_across_layers(model, tensors, layers, zero_type='random', share=0.25):
    for layer in layers:
        model = break_attn_heads_by_layer(model, tensors, zero_type, share, layer)
    return model
        

model = BertModel.from_pretrained('bert-base-uncased')
damaged_model = break_attn_across_layers(model, ['value', 'key'], [0, 2, 3, 4, 5])
