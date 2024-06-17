import inspect
import os
import torch
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
import time
import math

@dataclass
class gptConfig:
    def __init__(self,device) -> None:
        self.d_model = 768
        self.vocab_size = 50304
        self.n_layers = 12
        self.n_heads = 8
        self.context_length = 1024
        self.dropout = 0.2
        self.batch_size = 12
        self.device = device

class MultiHeadCausalAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.d_model, config.d_model*3)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.c_proj.flag = 1
        # Bad name for a causal mask(referred from the model published on HF), these ppl need to suffer(was questioning my existence trying understand why we use bias T.T)
        self.register_buffer('bias', torch.tril(torch.ones(config.context_length, config.context_length)).view(1, 1, config.context_length, config.context_length))
    def forward(self, x):
        B,T,C = x.size() # mistake 1 . used x.shape(), wasted 10 mins only to know compiler caNNNTT handle x.shape()
        d_model = self.config.d_model
        n_heads = self.config.n_heads
        qkv = self.c_attn(x)        
        # Splitting the attention heads
        # q = att[:,:,:d_model].view(bs, seq_length, n_heads, d_model//n_heads).transpose(1,2)
        # k = att[:,:,d_model:d_model*2].view(bs, seq_length, n_heads, d_model//n_heads).transpose(1,2)
        # v = att[:,:,d_model*2:d_model*3].view(bs, seq_length, n_heads, d_model//n_heads).transpose(1,2)
        q, k, v = qkv.split(self.config.d_model, dim=2)
        # Multiplying the query and key and scaling it
        if self.config.device == 'cpu': # very bad practice of including device in config
            att_val = (q @ k.transpose(-2,-1)) / (d_model ** 0.5)
            # Masking the attention values
            att_val = att_val.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att_val = F.softmax(att_val, dim=-1)
            att_val = att_val @ v

        #   Flash Attention
        if self.config.device == 'cuda':
            att_val = F.scaled_dot_product_attention(q, k, v, is_causal = True)

        # Multiplying the attention values with the value
        attention = (att_val).transpose(1,2).contiguous().view(B, T, C)
        final = self.c_proj(attention)
        return final
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_model*4)
        self.c_proj = nn.Linear(config.d_model*4, config.d_model)
        self.c_proj.flag = 1
        self.gelu = nn.GELU(approximate='tanh')
    def forward(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadCausalAttention(config)
        self.mlp = FeedForwardNetwork(config)
    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class gpt2(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.context_length, config.d_model),
            h = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.d_model),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        # sharing weights between the token embeddings and the final linear layer
        self.transformer['wte'].weight = self.lm_head.weight

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # for every residual connection we scale the weights by 1/sqrt(2*n_layers) so as to prevent the variance go boom
            if hasattr(module, 'flag') and module.flag == 1:
                std *= (2 * self.config.n_layers) ** -0.5
            module.weight.data.normal_(mean = 0.0, std = std)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean = 0.0, std = 0.02)

    def forward(self, x, targets = None):
        _, seq_length = x.size()
        assert seq_length <= self.config.context_length, "Input sequence length exceeds the context length"

        x = self.transformer['wte'](x) + self.transformer['wpe'](torch.arange(x.size(1)).to(device))
        for block in self.transformer['h']:
            x = block(x)
        x = self.transformer['ln_f'](x)
        x = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
        return x, loss
    
    def sample(self, ids, tokenizer):
        self.eval()
        with torch.no_grad():
            for i in range(100):
                out = self.forward(ids)[0]
                out = out[:,-1,:]
                probs = F.softmax(out, dim=-1)
                # sorts the probabilities and returns the indices of the sorted probabilities
                probs,indices = torch.topk(probs, 50, dim = -1)
                # sampling the next token from the top 50 tokens
                idx = torch.multinomial(probs, num_samples = 1)
                # since we sorted(kinda) the probabilities we need to get actual index of the token so we search the indices vec for index 'ix'
                actual_ids = torch.gather(indices, dim = -1, index = idx)
                ids = torch.cat([ids, actual_ids], dim = 1)
        ids = ids.view(-1).tolist()
        # print(len(ids))
        return ids
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def configure_optimizer(self, weight_decay, lr, device):

        param_dict = { x : y for x,y in self.named_parameters() }
        param_dict = { x : y for x,y in param_dict.items() if y.requires_grad}

        weight_decay = 0.1
        # separate the parameters for weight decay and no weight decay( bias and LayerNorm dont need weight decay)
        weight_decay_params = [y for x,y in param_dict.items() if y.dim()>=2]
        no_weight_decay_params = [y for x,y in param_dict.items() if y.dim()<2]
        
        optim_groups =[
            {'params': weight_decay_params, 'weight_decay': weight_decay},
            {'params': no_weight_decay_params, 'weight_decay': 0.0}
        ]
        decayed_params = sum(p.numel() for p in weight_decay_params)
        non_decayed_params = sum(p.numel() for p in no_weight_decay_params)
        print(f'Weight decay parameters: {decayed_params} Non weight decay parameters: {non_decayed_params}')
        fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        fused_bool = fused and device=='cuda'
        opt = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused = fused_bool)
        return opt

    

import tiktoken

tokenizer = tiktoken.get_encoding('gpt2')

def tokeinze_text(file):
    with open(file, 'r') as f:
        text = f.read()
    return tokenizer.encode(text)


class miniDataLoader:
    def __init__(self, config, data, process_rank, num_processes) -> None:
        self.bs = config.batch_size
        self.length = config.context_length
        self.data = data
        self.process_rank = process_rank
        self.start = self.bs * self.length * self.process_rank
        self.num_processes = num_processes
    def get_data(self):
        B = self.bs
        T = self.length
        if self.start + B*T*self.num_processes + 1 > len(self.data):
            self.start = B*T*self.process_rank
        d = self.data[self.start:self.start + B*T + 1]
        x = d[:-1].view(B,T)
        y = d[1:].view(B,T)
        self.start += B*T*self.num_processes
        return x,y

#--------------------------------------Distributed Data Parallel-----------------------------------------------------------------------------------------------

    #Optional Block, use for training the model on multiple GPU's using DDP

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:

    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ.get('RANK'))
    ddp_world_size = int(os.environ.get('WORLD_SIZE'))
    ddp_local_rank = int(os.environ.get('LOCAL_RANK'))
    device = f'cuda :{ddp_local_rank}'
    torch.cuda.set_device(device)
    master = ddp_rank == 0

else:
    ddp_rank = 0
    ddp_world_size = 1
    ddp_local_rank = 0
    msater = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using {device}")
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

B = 12
T = 1024
max_batch_size = 524288
validation_step = 100
# each every gpu runs its own process, so we need to divide the batch size by the number of gpu's
grad_accum_steps = max_batch_size // (B * T * ddp_world_size)
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    
    if it>max_steps:
        return min_lr
    
    decay_rate = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_rate <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_rate)) 
    return min_lr + coeff * (max_lr - min_lr)

# Training data
train_data = tokeinze_text('train.txt')
train_data = torch.tensor(train_data)
train_data_loader = miniDataLoader(gptConfig(device = device), train_data, ddp_rank, ddp_world_size)
# Validation data
val_data = torch.tensor(tokeinze_text('valid.txt'))
val_data_loader = miniDataLoader(gptConfig(device = device), val_data, ddp_rank, ddp_world_size)

model = gpt2(gptConfig(device = device))
print(model.count_parameters())
model.to(device) 
model = torch.compile(model, fullgraph = True, mode = 'reduce-overhead')

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
model_clone = model.module if ddp else model

torch.set_float32_matmul_precision('high')

def run_val(model, data_loader):
    model.eval()
    cummulative_loss = 0
    for i in range(10):
        x,y = data_loader.get_data()
        x,y = x.to(device), y.to(device)
        with torch.no_grad():
            out, loss = model(x,y)
        cummulative_loss += loss.detach()
    return cummulative_loss

optimizer = model_clone.configure_optimizer(0.1, 3e-4, device)

scaler = torch.cuda.amp.GradScaler()

for i in range(5000):
    
    optimizer.zero_grad()
    t1 = time.time()
    cummulative_loss = 0
    for steps in range(grad_accum_steps):
        x,y = train_data_loader.get_data()
        x,y = x.to(device), y.to(device)
    
        with torch.cuda.amp.autocast():
            out, loss = model_clone(x,y)
        loss = loss / grad_accum_steps
        cummulative_loss += loss.detach()
        # We wait for all models to calculate the loss and then we average the loss and deposit the average loss on each model nad backpropagate
        if ddp:
            model_clone.require_backward_grad_sync = steps == grad_accum_steps - 1
        loss.backward()
    if ddp:
        dist.all_reduce(cummulative_loss, op = dist.ReduceOp.AVG)  
    norm = torch.nn.utils.clip_grad_norm_(model_clone.parameters(), 1.0)
    if i%100 == 0:
            val_loss = run_val(model_clone, val_data_loader)
            if ddp:
                dist.all_reduce(val_loss, op = dist.ReduceOp.AVG)
            print(f'Validation loss: {val_loss}')
            text = model_clone.sample(torch.zeros(1,1).long().to(device), tokenizer)
            print(f'Sample: {tokenizer.decode(text)}')
            # import sys;sys.exit(0)
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t2 = time.time()
    print(f' Step: {i} Loss: {cummulative_loss:.6f} Time: {(t2-t1)*1000 :.4f} ms  Norm : {norm} LR: {lr}')

if ddp:
    destroy_process_group()

