import torch.nn as nn
import torch as torch
import tiktoken
from collections.abc import Iterator
import torch
import numpy as np
import math
import sys
import joblib
import datasets
from joblib import Parallel, delayed, parallel
from joblib.parallel import BatchCompletionCallBack
import regex
from collections import Counter
import pandas as pd
import datasets
from joblib import Parallel, delayed
from collections import Counter
from transformers import AutoTokenizer
from itertools import islice
from joblib import Parallel, delayed
from joblib.parallel import BatchCompletionCallBack
from joblib import parallel
from datasets import load_dataset
import pickle
import tiktoken
import copy
from copy import deepcopy
import torch
import einops
from einops import rearrange, einsum
import torch.nn as nn
from transformers import AutoTokenizer
from itertools import islice
import torch.nn.functional as F
from tokenizers import decoders
from tokenizers import Tokenizer

BatchStream = Iterator[
    tuple[torch.Tensor, torch.Tensor]
]



class Linear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        self.d_model = in_features
        # we assume the output has dimensions batch_size, d_out
        self.d_out = out_features
        if dtype is None:
            self.dtype2 = torch.float32
        if dtype is not None:
            self.dtype2 = dtype
        #initializing W  = self.weights
        self.weights = nn.Parameter(torch.empty(self.d_out, self.d_model, dtype = self.dtype2, device=device) )
        self.sigma = np.sqrt(2/(self.d_model + self.d_out))
        torch.nn.init.trunc_normal_(self.weights, mean=0.0, std= self.sigma, a=-3*self.sigma,
        b=3*self.sigma)


    def forward(self, x: torch.Tensor)->torch.Tensor:
        x  = x.to(dtype=self.weights.dtype, device=self.weights.device)
        #return x @ self.weights.T
        return einsum(x, self.weights, "... dmodel,  dout dmodel -> ... dout")
    
'''
We now implement the embedding module.
'''

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
        super().__init__()
        if dtype is None:
            self.dtype2 = torch.float32
        if dtype is not None:
            self.dtype2 = dtype
        self.embedding_matrix = nn.Parameter(
        torch.empty(num_embeddings, embedding_dim, dtype = self.dtype2, device=device)
        )
        torch.nn.init.trunc_normal_(self.embedding_matrix, mean=0.0, std= 1, a = -3, b = 3)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids, :]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        if dtype is None:
            self.dtype2 = torch.float32
        if dtype is not None:
            self.dtype2 = dtype
        self.epsilon = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # here x is a tensor of type (batch_size, sequence_length, d_model)
        d_model = (x.shape)[-1] #dmodel is the last dimension of x
        x1 = torch.sum(torch.square(x), dim = -1, keepdim = True)
        rms = torch.sqrt(x1/d_model + self.epsilon)
        x2 = x/rms 
        return x2


'''
We now implement position-wise feed-forward network
'''
class FeedForward(nn.Module):
    def __init__(self, d_model, dtype = None, device = None):
        super().__init__()
        #here dff is the hidden dimension of the W matrices
        self.dff = (8/3)*d_model
        self.dff = math.ceil(self.dff/64)*64
        self.dff = int(self.dff)
        if dtype is None:
            self.dtype2 = torch.float32
        if dtype is not None:
            self.dtype2 = dtype
        self.W3 = Linear(d_model, self.dff, device=device)
        self.W1 = Linear(d_model, self.dff, device=device)
        self.W2 = Linear(self.dff, d_model, device=device)
    def forward(self, x: torch.Tensor)->torch.Tensor:
        silu=nn.SiLU()
        x = x.to(torch.float32)
        w3x = self.W3(x)
        w1x = self.W1(x)
        w1xp = silu(w1x)
        prod = einsum(w1xp, w3x, " ... dff, ... dff -> ... dff ")
        finalprod = self.W2(prod)
        return finalprod

'''We now implement RoPE'''
class RoPE_Embedding(nn.Module):
    def __init__(self, theta:float, d_k: int, max_seq_len: int, device = None):
        #we assume our feature vector dimension d_k to be even
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"
        self.dim_p = d_k // 2
        SinMat = torch.zeros(max_seq_len, self.dim_p, device = device)
        CosMat = torch.zeros(max_seq_len, self.dim_p, device = device)
        for i in range(0, max_seq_len):
            for k in range(0, self.dim_p):
                angle = i/theta**((2*k)/d_k)
                CosMat[i, k]= np.cos(angle)
                SinMat[i, k]= np.sin(angle)

        self.register_buffer("CMat", CosMat, persistent = False)  
        self.register_buffer("SMat", SinMat, persistent = False)             
                
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor)->torch.Tensor:
        # x has dimensions (..., seq_len, d_k). Here seq_len doesn't necessarily index the absolution positions
        # token_positions has dimensions (..., seq_len)
        pos = token_positions.long()
        cos = self.CMat[pos, :]
        sin = self.SMat[pos, :]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot_even = cos*x_even - sin*x_odd
        rot_odd = sin*x_even + cos*x_odd
        y = x.clone()
        y[..., 0::2] = rot_even
        y[..., 1::2] = rot_odd
        return y

'''We now implement the Phrase version of Rope'''

class RoPE_Embedding_phrasev2(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
        min_position: int = 0
    ):
        super().__init__()

        assert d_k % 2 == 0, "d_k must be even"

        self.dim_p = d_k // 2
        self.min_position = min_position
        self.max_seq_len = max_seq_len

        # Stored table indices:
        # 0, 1, ..., max_seq_len-1
        #
        # Corresponding physical positions:
        # min_position, min_position+1, ..., min_position+max_seq_len-1

        positions = torch.arange(
            min_position,
            min_position + max_seq_len,
            device=device,
            dtype=torch.float32
        )

        k = torch.arange(
            self.dim_p,
            device=device,
            dtype=torch.float32
        )

        # Same frequency convention as your original implementation
        freq = theta ** (-2 * k / d_k)

        angles = positions[:, None] * freq[None, :]

        self.register_buffer(
            "CMat",
            torch.cos(angles),
            persistent=False
        )

        self.register_buffer(
            "SMat",
            torch.sin(angles),
            persistent=False
        )

    def forward(
        self,
        x: torch.Tensor,
        position_indices: torch.Tensor
    ) -> torch.Tensor:

        # position_indices are TABLE indices:
        # 0, 1, ..., max_seq_len-1

        pos = position_indices.long()

        cos = self.CMat[pos, :]
        sin = self.SMat[pos, :]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rot_even = cos * x_even - sin * x_odd
        rot_odd = sin * x_even + cos * x_odd

        y = x.clone()
        y[..., 0::2] = rot_even
        y[..., 1::2] = rot_odd

        return y

'''We now implement softmax on a tensor using a function'''
def soft_max( x: torch.Tensor, dimen: int):
    if dimen > len(x.shape)-1:
        print("you are applying softmax to the wrong dimension")
    else:
        v_max = torch.max(x, dim = dimen, keepdim=True).values #max gives both values and indices where max occurs
        x  = x - v_max
        x_exp = torch.exp(x)
        x_sum = torch.sum(x_exp, dim = dimen, keepdim = True)
        softM = x_exp/x_sum
        return softM


'''We now implement scaled dot-product attention'''
def string_f1(a: list):
    x = ''
    for i in range(0, len(a)):
        if i == len(a) -1:
                x =x +chr(a[i]+97)
        else:
            x =x +chr(a[i]+97) + ' '
    return x

def string4einsum(a1:list, a2:list, a3:list):
    x = string_f1(a1) + ', ' + string_f1(a2) +  ' -> ' + string_f1(a3)
    return x


def MaskMatrix(x: torch.Tensor, token_positions: torch.Tensor):
    # x has dimensions (batch_size, ..., seq_len, d_model). Here seq_len doesn't necessarily index the absolution positions
    # token_positions has dimensions (batch_size..., seq_len)
    # Mask = torch.full(x.shape[0: x.dim()-1] + (x.shape[-1],), True) # i am excluding the last dimension of size d_model and then adding seq_len
    # mask has dimensions (batch_size, ..., sqlen1, sqlen2)
    Mask = token_positions[..., :, None] >= token_positions[..., None, :]
    
    return Mask



def attention_matrix( Q: torch.Tensor, K: torch.Tensor, mask: torch.Tensor):
    #here Q, K has shape (batch_size, ..., seq_len, d_k)
    # in this code, we compute the tilde S matrix or the attention matrix
    # here, S_oft, the output has dimensions (batch_size, ..., sqlen1, sqlen2)
    # here (batch_size, ...) are the batch-like dimensions and attention is computed only between the sqlen1, sqlen2 dimensions
    # ... since there is no attention between tokens with different batch-like dimensions
    d_k = Q.shape[-1]
    n1 = Q.dim()-2 # dimensions of everything except the last dim two of Q, K
    a1 = [i for i in range(1, n1+1)]
    totx = string4einsum(a1+ [n1+1, n1+2], a1+[n1+3, n1+2], a1+ [n1+1, n1+3])
    S = einsum(Q, K, totx)/np.sqrt(d_k)
    S[mask == False] = -float('inf') # step applies the mask M which says whether ith query should attend to jth key
    S_soft = soft_max(S, dimen = -1)
    
    return S_soft

def attention_matrix_interleave_helper( Q: torch.Tensor, K: torch.Tensor, mask: torch.Tensor):
    #here Q, K has shape (batch_size, ..., seq_len, d_k)
    # in this code, we compute the tilde S matrix or the attention matrix
    # here, S_oft, the output has dimensions (batch_size, ..., sqlen1, sqlen2)
    # here (batch_size, ...) are the batch-like dimensions and attention is computed only between the sqlen1, sqlen2 dimensions
    # ... since there is no attention between tokens with different batch-like dimensions
    d_k = Q.shape[-1]
    n1 = Q.dim()-2 # dimensions of everything except the last dim two of Q, K
    a1 = [i for i in range(1, n1+1)]
    totx = string4einsum(a1+ [n1+1, n1+2], a1+[n1+3, n1+2], a1+ [n1+1, n1+3])
    S = einsum(Q, K, totx)/np.sqrt(d_k)
    #S[mask == False] = -float('inf') # step applies the mask M which says whether ith query should attend to jth key
    # S_soft = soft_max(S, dimen = -1)
    
    return S

def attn_output(S: torch.Tensor, V: torch.tensor):
    # S is the attention matrix from above
    # V is the value matrix with dimensions (batch, ..., seq_len, d_v)
    # S has dimensions (batch, ..., seq_len, seq_len)
    n = V.dim() - 2 # batch-like dimensions
    a1 = [i for i in range(1, n+1)] + [n+1, n+2]
    a2 = [i for i in range(1, n+1)] + [n+2, n+3]
    a3 = [i for i in range(1, n+1)] + [n+1, n+3]
    xtot = string4einsum(a1 , a2 , a3 )
    out = einsum(S, V, xtot)
    return out

'''We now write code for phrase transformers'''
# I now write useful functions for local phrase transformer:

def PAttnHelper(token_positions: torch.Tensor, phrase_len: int):
    # token_positions: (..., s)
    k = phrase_len
    # (..., s, 1)
    positions = token_positions.unsqueeze(-1)
    # [0, 1, ..., k]
    offsets = torch.arange(
        k + 1,
        device=token_positions.device
    )
    # (..., s, k+1)
    r = positions - k + offsets
    # Valid positions cannot be negative
    mask = r >= 0
    # Clamp padded entries to 0 so they are safe indices into K
    r = r.clamp_min(0)
    return r, mask

def phr_attention_matrix(Q: torch.Tensor, K : torch.Tensor, r: torch.Tensor, mask: torch.Tensor):
    # Q/K is the query, key value matrix with dimensions (batch_size, ..., seq, dk)
    # token_positions has dimensions (batch_size, ..., seq) 
    # the output has dimensions (batch_size, ..., seq, phrL+1)
    # r, mask has dimensions (batch_size, ..., seq, phr_len+1)
    dk = Q.shape[-1]
    P = r.shape[-1]
    d = K.shape[-1]

    # K_expanded: (..., seq, P, dk)
    K_expanded = K.unsqueeze(-2).expand(*K.shape[:-1], P, d)
    # r_expanded: (..., seq, P, dk)
    r_expanded = r.unsqueeze(-1).expand(*r.shape, d)
    # (..., seq, P, dk)
    K_sel = torch.gather(K_expanded, dim=-3, index=r_expanded)
    S = torch.einsum('...sj,...sij->...si', Q, K_sel)
    S = S / np.sqrt(dk)
    S = S.masked_fill(~mask, -float("inf"))
    return torch.softmax(S, dim=-1)

def phr_attn_output(S: torch.Tensor, V: torch.Tensor, r: torch.Tensor, mask: torch.Tensor):
    # S has dimensions (batch_size, ..., seq, p_L)
    # V has dimensions (batch_size, ..., seq, dv)
    # r, mask has dimensions (batch_size, ..., seq, phr_len+1)
    # the output should dimensions (batch_size, ..., seq, dv)
    d = V.shape[-1]
    P = r.shape[-1]

    V_expanded = V.unsqueeze(-2).expand(*V.shape[:-1], P, d)
    r_expanded = r.unsqueeze(-1).expand(*r.shape, d)
    V_sel = torch.gather(
    V_expanded,
    dim=-3,
    index=r_expanded
    )
    out = torch.einsum(
    '...si,...sid->...sd',
    S,
    V_sel
    )
    return out

class phrase_causal_multi_head_attn(nn.Module):
    def __init__(self, d_model, num_heads, theta_phrase, phr_len, device=None):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % 2 == 0
        self.dk = self.dv = d_model // num_heads
        self.num_heads, self.phr_len = num_heads, phr_len

        self.Wk_list = nn.ModuleList([Linear(d_model, self.dk, device=device) for _ in range(num_heads)])
        self.Wq_list = nn.ModuleList([Linear(d_model, self.dk, device=device) for _ in range(num_heads)])
        self.Wv_list = nn.ModuleList([Linear(d_model, self.dv, device=device) for _ in range(num_heads)])
        self.Wout_phrase = Linear(num_heads * self.dv, d_model, device=device)
        self.rope_phrase = RoPE_Embedding_phrasev2(theta_phrase, self.dk, phr_len + 1, device=device,
                                                   min_position=-phr_len)

    def forward(self, x, token_positions):
        Q = [Wq(x) for Wq in self.Wq_list]
        K = [Wk(x) for Wk in self.Wk_list]
        V = [Wv(x) for Wv in self.Wv_list]
        r, mask = PAttnHelper(token_positions, self.phr_len)
        P, outs = self.phr_len + 1, []

        idx = torch.arange(P, device=x.device)
        cos, sin = self.rope_phrase.CMat[idx], self.rope_phrase.SMat[idx]

        for i in range(self.num_heads):
            q, k = Q[i], K[i]
            d = k.shape[-1]

            k_exp = k.unsqueeze(-2).expand(*k.shape[:-1], P, d)
            r_exp = r.unsqueeze(-1).expand(*r.shape, d)
            k_sel = torch.gather(k_exp, -3, r_exp)

            ke, ko = k_sel[..., 0::2], k_sel[..., 1::2]
            k_rot = torch.empty_like(k_sel)
            k_rot[..., 0::2] = cos * ke - sin * ko
            k_rot[..., 1::2] = sin * ke + cos * ko

            S = torch.einsum("...nd,...nld->...nl", q, k_rot) / np.sqrt(self.dk)
            S = torch.softmax(S.masked_fill(~mask, -float("inf")), dim=-1)

            outs.append(phr_attn_output(S, V[i], r, mask))

        return self.Wout_phrase(torch.cat(outs, dim=-1))

class PhraseTransBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta_phrase: float, phr_len:int, device = None):
        super().__init__()
        self.attn = phrase_causal_multi_head_attn(d_model, num_heads, theta_phrase, phr_len, device = device)
        self.d_model = d_model
        self.RMSNormA = RMSNorm(d_model, 1e-5, device = device)
        # self.FeedForward_normal = FeedForward(d_model, device = device)
        self.FeedForward_phrase = FeedForward(d_model, device = device)
        # self.FeedForwardCross = FeedForwardCross(2*d_model, device = device)
        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        # let x have dimensions (batch_size, ..., seq_len, d_model)
        # let token_positions have dimensions (batch_size, ..., position)
        diagnostics = {}
        y = x.clone()
        y = self.RMSNormA(y)
        output_attn_phrase = self.attn(y, token_positions) # this has dimensions (batch_size, ..., seq_len, d_model)
        # z1_normal = x + output_attn_normal #this is the output of the first part of the transformer block
        z1_phrase = x + output_attn_phrase #...
        # z2_normal= z1_normal.clone()
        z2_phrase = z1_phrase.clone()
        # z2_normal = self.RMSNormA(z2_normal)
        z2_phrase = self.RMSNormA(z2_phrase)
        # FeedForwardOutput_normal = self.FeedForward_normal(z2_normal) # this has dimensions (batch_size, ..., seq_len, d_model)
        FeedForwardOutput_phrase = self.FeedForward_phrase(z2_phrase)
        # FeedForwardOutput_cross  = self.FeedForwardCross(torch.cat([z2_normal, z2_phrase], -1))
        # diagnostics["input_rms"] = tensor_rms(x).item()

        # diagnostics["attn_normal_rms"] = tensor_rms(output_attn_normal).item()
        # diagnostics["attn_phrase_rms"] = tensor_rms(output_attn_phrase).item()

        # diagnostics["ffn_normal_rms"] = tensor_rms(FeedForwardOutput_normal).item()
        # diagnostics["ffn_phrase_rms"] = tensor_rms(FeedForwardOutput_phrase).item()
        # diagnostics["ffn_cross_rms"] = tensor_rms(FeedForwardOutput_cross).item()
        
        total_update = (
            # output_attn_normal
            output_attn_phrase
            # + FeedForwardOutput_normal
            + FeedForwardOutput_phrase
            # + FeedForwardOutput_cross
        )
        # diagnostics["total_update_rms"] = tensor_rms(total_update).item()
        
        z3 = x + total_update #this is the final output of the transformer block
        # diagnostics["cross_fraction"] = (
        #     tensor_rms(FeedForwardOutput_cross)
        #     / (tensor_rms(total_update) + 1e-8)
        # ).item()
        return z3 
    
class PhraseTransformerLM(nn.Module):
    def __init__(self, vocab_size:int, d_model: int, num_heads: int,  n_layers:int, theta_phrase: float,  phr_len: int, device = None):
        super().__init__()
        self.TransBlockList = nn.ModuleList([PhraseTransBlock(d_model, num_heads, theta_phrase, phr_len, device = device) for _ in range(n_layers)])
        self.RMSNormA = RMSNorm(d_model, 1e-5, device = device)
        self.Wlogits = Linear(d_model, vocab_size, device = device)
        self.Embedding  = Embedding(vocab_size, d_model, device = device)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = n_layers
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        # y -> (batch_size, ..., sq_len, d_model) contains token ids
        # token_positions -> (batch_size, ..., position)
        # logits has dimension (batch_size, ..., seq_len, vocab_size)
        x = self.Embedding(x)
        all_diagnostics = []
        for i in range(self.num_layers):
            t = self.TransBlockList[i]
            x  = t(x, token_positions)
            # diagnostics["layer"] = i
            # all_diagnostics.append(diagnostics) 
        x = self.RMSNormA(x)
        logits = self.Wlogits(x)
        #return soft_max(logits, -1)
        return logits 
    def hidden(self, x, token_positions):
        x = self.Embedding(x)
        # the out dimensions are (batch_size, ..., d_model)
        for i in range(self.num_layers):
            t = self.TransBlockList[i]
            x  = t(x, token_positions)
            # diagnostics["layer"] = i
            # all_diagnostics.append(diagnostics) 
        x = self.RMSNormA(x)
        return x 

'''We now write code for normal transformers'''
class causal_multi_head_attn(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.dk = d_model//num_heads # dimensions of key/query vectors
        self.dv = d_model//num_heads # dimensions of value vectors
        self.num_heads = num_heads

        self.Wk_list = nn.ModuleList([Linear(d_model, self.dk, device=device) for _ in range(0, num_heads)])
        self.Wq_list = nn.ModuleList([Linear(d_model, self.dk, device=device) for _ in range(0, num_heads)])
        self.Wv_list = nn.ModuleList([Linear(d_model, self.dv, device=device) for _ in range(0, num_heads)])
        self.Wout = Linear(num_heads*self.dv, d_model, device=device)
        #initializing rope for encoding positional intormation
        self.rope  = RoPE_Embedding(theta, self.dk, max_seq_len, device=device)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor ):
        Q_matrices = [Wq(x) for Wq in self.Wq_list]
        K_matrices = [Wk(x) for Wk in self.Wk_list]
        V_matrices = [Wv(x) for Wv in self.Wv_list] #initialized the Q_i, K_i, V_i
        Q_matrices = [self.rope(q, token_positions) for q in Q_matrices]
        K_matrices = [self.rope(k ,token_positions) for k in K_matrices]

        Mask = MaskMatrix(x, token_positions) # computes the mask matrix
        attn_matrices = [attention_matrix(Q_matrices[i], K_matrices[i], Mask) for i in range(0, self.num_heads)]
        Out_matrices = [attn_output(attn_matrices[i], V_matrices[i]) for i in range(0, self.num_heads)]
        Out_matrices_cat = torch.cat(Out_matrices, dim = -1)
        Output_matrices2 = self.Wout(Out_matrices_cat)
        return Output_matrices2 #This concludes the attention mechanism
    
class TransBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device = None):
        super().__init__()
        self.attn = causal_multi_head_attn(d_model, num_heads, theta, max_seq_len, device = device)
        self.d_model = d_model
        self.RMSNormA = RMSNorm(d_model, 1e-5, device = device)
        self.FeedForwardA = FeedForward(d_model, device = device)
        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        # let x have dimensions (batch_size, ..., seq_len, d_model)
        # let token_positions have dimensions (batch_size, ..., position)
        y = x.clone()
        y = self.RMSNormA(y)
        output_attn = self.attn(y, token_positions) # this has dimensions (batch_size, ..., seq_len, d_model)
        z1 = x + output_attn #this is the output of the first part of the transformer block
        z2= z1.clone()
        z2 = self.RMSNormA(z2)
        FeedForwardOutput = self.FeedForwardA(z2) # this has dimensions (batch_size, ..., seq_len, d_model)
        z3 = z1 + FeedForwardOutput #this is the final output of the transformer block
        return z3
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size:int, d_model: int, num_heads: int, n_layers:int, theta: float, max_seq_len: int, device = None):
        super().__init__()
        self.TransBlockList = nn.ModuleList([TransBlock(d_model, num_heads, theta, max_seq_len, device = device) for _ in range(n_layers)])
        self.RMSNormA = RMSNorm(d_model, 1e-5, device = device)
        self.Wlogits = Linear(d_model, vocab_size, device = device)
        self.Embedding  = Embedding(vocab_size, d_model, device = device)
        self.vocab_size = vocab_size
        self.d_model = d_model
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        # y -> (batch_size, ..., sq_len, d_model) contains token ids
        # token_positions -> (batch_size, ..., position)
        # logits has dimension (batch_size, ..., seq_len, vocab_size)
        x = self.Embedding(x)
        for t in self.TransBlockList:
            x = t(x, token_positions)
        x = self.RMSNormA(x)
        logits = self.Wlogits(x)
        #return soft_max(logits, -1)
        return logits


'''We now write code for the dilated transformer'''

def DilatedGlobalMask(
    token_positions: torch.Tensor,
    phrase_len: int,
    stride: int,
):
    """
    token_positions:
        (batch, ..., seq_len)

    Returns:
        (batch, ..., query_seq_len, key_seq_len)

    The global stream attends to:
        1. The current token itself.
        2. Long-range past tokens spaced by `stride`.
        3. No future tokens.
        4. No nearby past tokens handled by the phrase stream.
    """
    if stride <= 0:
        raise ValueError("stride must be positive")
    query_positions = token_positions[..., :, None]
    key_positions = token_positions[..., None, :]
    distance = query_positions - key_positions
    causal = distance >= 0
    self_connection = distance == 0
    dilated_long_range = (
        (distance > phrase_len)
        & (distance.remainder(stride) == 0)
    )
    mask = causal & (
        self_connection | dilated_long_range
    )
    return mask    


class causal_multi_head_attn_dilated(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, stride: int, phr_len: int, device = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.dk = d_model//num_heads # dimensions of key/query vectors
        self.dv = d_model//num_heads # dimensions of value vectors
        self.num_heads = num_heads

        self.Wk_list = nn.ModuleList([Linear(d_model, self.dk, device=device) for _ in range(0, num_heads)])
        self.Wq_list = nn.ModuleList([Linear(d_model, self.dk, device=device) for _ in range(0, num_heads)])
        self.Wv_list = nn.ModuleList([Linear(d_model, self.dv, device=device) for _ in range(0, num_heads)])
        self.Wout = Linear(num_heads*self.dv, d_model, device=device)
        #initializing rope for encoding positional intormation
        self.rope  = RoPE_Embedding(theta, self.dk, max_seq_len, device=device)
        self.phr_len = phr_len
        self.stride = stride

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor ):
        Q_matrices = [Wq(x) for Wq in self.Wq_list]
        K_matrices = [Wk(x) for Wk in self.Wk_list]
        V_matrices = [Wv(x) for Wv in self.Wv_list] #initialized the Q_i, K_i, V_i
        Q_matrices = [self.rope(q, token_positions) for q in Q_matrices]
        K_matrices = [self.rope(k ,token_positions) for k in K_matrices]
        
        Mask = DilatedGlobalMask(token_positions, self.phr_len, self.stride)

        # Mask = MaskMatrix(x, token_positions) # computes the mask matrix
        attn_matrices = [attention_matrix(Q_matrices[i], K_matrices[i], Mask) for i in range(0, self.num_heads)]
        Out_matrices = [attn_output(attn_matrices[i], V_matrices[i]) for i in range(0, self.num_heads)]
        Out_matrices_cat = torch.cat(Out_matrices, dim = -1)
        Output_matrices2 = self.Wout(Out_matrices_cat)
        return Output_matrices2 #This concludes the attention mechanism
    
class TransBlockDilated(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, stride: int, phr_len: int, device = None):
        super().__init__()
        self.attn = causal_multi_head_attn_dilated(d_model, num_heads, theta, max_seq_len, stride, phr_len, device = device)
        self.d_model = d_model
        self.RMSNormA = RMSNorm(d_model, 1e-5, device = device)
        self.FeedForwardA = FeedForward(d_model, device = device)
        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        # let x have dimensions (batch_size, ..., seq_len, d_model)
        # let token_positions have dimensions (batch_size, ..., position)
        y = x.clone()
        y = self.RMSNormA(y)
        output_attn = self.attn(y, token_positions) # this has dimensions (batch_size, ..., seq_len, d_model)
        z1 = x + output_attn #this is the output of the first part of the transformer block
        z2= z1.clone()
        z2 = self.RMSNormA(z2)
        FeedForwardOutput = self.FeedForwardA(z2) # this has dimensions (batch_size, ..., seq_len, d_model)
        z3 = z1 + FeedForwardOutput #this is the final output of the transformer block
        return z3


class causal_multi_head_cross_attn(nn.Module):
    def __init__(self, d_model_phrase: int, d_model_global: int, num_heads_global: int, theta_global: float, max_seq_len: int, device = None):
        # let x be the inputs/outputs in the phrase stream
        # let y be the inputs/outputs in the global stream
        #         globalout:
        #     Q from global
        #     K/V from phrase
        #     output updates global

        # phraseout:
        #     Q from phrase
        #     K/V from detached global
        #     output updates phrase
        
        super().__init__()
        assert d_model_global % num_heads_global == 0
        # assert d_model_phrase % num_heads_phrase == 0
        self.dk_global = d_model_global//num_heads_global # dimensions of key/query vectors
        self.dv_global = d_model_global//num_heads_global # dimensions of value vectors
        self.num_heads_global = num_heads_global
        assert self.dk_global%2 == 0
    
        self.Wq_globalout_list = nn.ModuleList([Linear(d_model_global, self.dk_global, device=device) for _ in range(0, num_heads_global)])
        self.Wk_globalout_list = nn.ModuleList([Linear(d_model_phrase, self.dk_global, device=device) for _ in range(0, num_heads_global)])
        self.Wv_globalout_list = nn.ModuleList([Linear(d_model_phrase, self.dv_global, device=device) for _ in range(0, num_heads_global)])

        #initializing rope for encoding positional intormation
        self.rope_global  = RoPE_Embedding(theta_global, self.dk_global, max_seq_len, device=device)
        self.Wout_global =  Linear(num_heads_global*self.dv_global, d_model_global, device=device)  

    def forward(self, x: torch.Tensor, y: torch.Tensor, token_positions: torch.Tensor):
        # x is from the phrase stream, has dimensions (batch_size, ..., seq_len, d_model_phrase)
        # y is from the global stream, has dimensions (batch_size, ..., seq_len, d_model_global)
        Q_matrices_global = [Wq(y) for Wq in self.Wq_globalout_list]
        K_matrices_global = [Wk(x) for Wk in self.Wk_globalout_list]
        V_matrices_global = [Wv(x) for Wv in self.Wv_globalout_list] #initialized the Q_i, K_i, V_i

        Q_matrices_global = [self.rope_global(q, token_positions) for q in Q_matrices_global]
        K_matrices_global = [self.rope_global(k, token_positions) for k in K_matrices_global]
        # Q_matrices_phrase = [self.rope_phrase(q, token_positions) for q in Q_matrices_phrase]
        # K_matrices_phrase = [self.rope_phrase(k, token_positions) for k in K_matrices_phrase]
        
        Mask = MaskMatrix(x , token_positions) # computes the mask matrix for normal attention
        attn_matrices_global =  [attention_matrix(Q_matrices_global[i], K_matrices_global[i], Mask) for i in range(self.num_heads_global)]
        out_matrices_global =  [attn_output(attn_matrices_global[i], V_matrices_global[i]) for i in range(self.num_heads_global)]    
        out_matrices_global = torch.cat(out_matrices_global, dim = -1)

        return self.Wout_global(out_matrices_global)  #this has dimensions ( batch, ..., seq_len, d_model)


class CrossTransBlock(nn.Module):
    def __init__(self, d_model_phrase: int, d_model_global: int, num_heads_global: int, theta_global: float, max_seq_len: int, device = None ):
        super().__init__()
        self.crossattn = causal_multi_head_cross_attn( d_model_phrase, d_model_global, num_heads_global, theta_global, max_seq_len, device= device)
        self.RMSNormGlobal = RMSNorm(d_model_global, 1e-5, device = device)
        self.RMSNormPhrase = RMSNorm(d_model_phrase, 1e-5, device = device)
        self.FeedForward_global = FeedForward(d_model_global, device = device)
        # self.FeedForward_phrase = FeedForward(d_model_phrase, device = device)

    def forward(self, x: torch.Tensor, y: torch.Tensor, token_positions: torch.Tensor, coup: float):
        # x has dimensions (batch_size, ..., sq_len, d_model_phrase)
        # y has dimensions (batch_size, ..., sq_len, d_model_global)
        xc = x.clone()
        yc = y.clone()
        xc = self.RMSNormPhrase(x)
        yc = self.RMSNormGlobal(y)
        attn_out_global = self.crossattn(xc, yc, token_positions)
        y1 = y + coup*attn_out_global
        y2 = y1.clone()
        y2 = self.RMSNormGlobal(y2)
        FFNy = self.FeedForward_global(y2)
        y3 = y1 + FFNy

        return y3
   
def tensor_rms(x):
    return torch.sqrt(torch.mean(x.detach() ** 2))
    
 
    

class NewTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model_phrase: int, d_model_global: int, num_heads_phrase: int, num_heads_global: int, num_heads_cross: int, n_layers_phrase: int, n_layers_global: int, theta_normal: float, theta_phrase: float, theta_cross: float, max_seq_len: int, phr_len: int, stride: int, num_cross_block: int, device = None):
        super().__init__()
        self.TransBlockList =  nn.ModuleList( [TransBlockDilated( d_model_global, num_heads_global, theta_normal, max_seq_len, stride, phr_len, device = device) for _ in range(n_layers_global)] )
        self.PhraseTransBlockList =  nn.ModuleList( [PhraseTransBlock(d_model_phrase, num_heads_phrase, theta_phrase, phr_len, device = device) for _ in range(n_layers_phrase)] )
        self.CrossTransList = nn.ModuleList( [CrossTransBlock( d_model_phrase, d_model_global, num_heads_cross, theta_cross, max_seq_len, device = device) for _ in range(num_cross_block)] )
        self.RMSNorm_global = RMSNorm(d_model_global, 1e-5, device = device)
        self.Wlogits_global= Linear(d_model_global, vocab_size, device = device)
        self.Embedding_global  = Embedding(vocab_size, d_model_global, device = device)
        self.Embedding_phrase  = Embedding(vocab_size, d_model_phrase, device = device)
        self.vocab_size = vocab_size
        self.d_model_global = d_model_global
        self.d_model_phrase = d_model_phrase
        self.num_layers_phrase = n_layers_phrase
        self.num_layers_global = n_layers_global
        self.num_cross_block = num_cross_block
        list1 = range(self.num_layers_phrase)
        list2 = range(self.num_layers_global)
        assert num_cross_block > 0
        assert num_cross_block <= n_layers_phrase
        assert num_cross_block <= n_layers_global
        self.parts_phrase = [part.tolist() for part in np.array_split(list1, self.num_cross_block)]   
        self.parts_global = [part.tolist() for part in np.array_split(list2, self.num_cross_block)]
        # self.d_model_input = d_model_input
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor, coup: float):
        # x has tokens with dimensions (batch_size, ..., sq_len). These are the input tokens
        # token_positions (batch_size, ..., sq_len)
        # logits has dimensions (batch_size, ..., seq_len, vocab_len)
        y = self.Embedding_global(x) # y is the embedding for the global stream 
        x = self.Embedding_phrase(x) # x is the embedding for the phrase stream
        

        for i in range(self.num_cross_block):
            phrase_list = self.parts_phrase[i]
            global_list = self.parts_global[i]
            #implementing normal transformers in both phrase and global streams
            for k in phrase_list:
                phrase_trans = self.PhraseTransBlockList[k]
                x = phrase_trans(x, token_positions)
            for k in global_list:
                global_trans = self.TransBlockList[k]
                y = global_trans(y, token_positions)
            #implementing cross attention 
            cross_trans = self.CrossTransList[i]
            y = cross_trans(x, y, token_positions, coup)
        # x = self.RMSNorm_phrase(x)
        y = self.RMSNorm_global(y)
        # logits_phrase = self.Wlogits_phrase(x)
        logits_global = self.Wlogits_global(y)
        return logits_global




# We now implement the AdamW optimizer
class AdamWOpt(torch.optim.Optimizer):
    def __init__(self, params, tinit = 0, alpha = 1e-4, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, lambda_decay = 0.01):
        defaults = {"t": tinit ,"alpha": alpha, "beta1": beta1, "beta2": beta2, "eps": eps, "lambda_decay": lambda_decay}
        super().__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            alpha = group["alpha"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            lambda_decay = group["lambda_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # here since self.state is a default(dict), it can be initialized doing this
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.data)          # initializing first moment
                    state["v"] = torch.zeros_like(p.data)          #  initializing second moment
                    state["t"] = 1
                t = state["t"]              
                alpha_t = alpha*math.sqrt(1-beta2**t)/(1-beta1**t)
                grad = p.grad.data
                p.data -= alpha*lambda_decay*p.data             # updating parameter weights
                m = state["m"]                                  # assigning m
                v = state["v"]                                  # assigning v
                m = beta1*m + (1-beta1)*grad                       # updating first moment 
                v = beta2*v + (1-beta2)*grad**2                    # updating second moment
                state["m"] = m                                  # storing new m
                state["v"] = v                                  # storing new v
                p.data -= alpha_t*m/(torch.sqrt(v)+eps)
                state["t"] = t+1                                   # t is t+1
        return None

# Learning schedule
# def LearningSched(eta_max, eta_min, T_w, T_tot, t):
#     if t <= T_w:
#         return eta_max*np.sin((np.pi*t)/(2*T_w))
#     if T_w < t <= T_tot:
#         return eta_min + (1/2)*(eta_max - eta_min)*(1+np.cos(np.pi*(t-T_w)/(T_tot - T_w)))
#     if t > T_tot:
#         return LearningSched(eta_max, eta_min, T_w, T_tot, t = T_tot)
def eta(eta_min, eta_max, T_w, T, t):
    return eta_min+((eta_max - eta_min)/2) * (
        1 + np.cos(
            np.pi * (t - T_w) / (T - T_w)
        )
    )


def eta_schedule(eta_max,eta_min, T_w, T,alpha,beta,t,):
    if t <= T_w:
        return eta_max * np.sin(np.pi * t / (2 * T_w))
    if T_w < t <= alpha * T:
        return eta_max
    if alpha*T < t <=T:
        return eta(eta_min,eta_max,alpha * T,beta * T,t)
    if t> T:
        return eta(eta_min,eta_max,alpha * T,beta * T,T)
    


'''This function takes in data and produces batches'''
def token_batch_stream(
    train_iter,
    enc,
    seq_len,
    batch_size,
    documents_per_tokenization=64,
):
    buffer = []
    start = 0

    # One extra token is required for shifted targets.
    block_size = seq_len + 1
    tokens_per_batch = batch_size * block_size

    while True:
        examples = list(
            islice(train_iter, documents_per_tokenization)
        )

        if not examples:
            break

        texts = [
            example["text"].strip()
            for example in examples
            if example["text"].strip()
        ]

        if not texts:
            continue

        # Fast tokenizers process a list more efficiently.
        encoded_documents = enc(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=False,
            verbose=False,
        )["input_ids"]

        for document_tokens in encoded_documents:
            buffer.extend(document_tokens)
            buffer.append(enc.eos_token_id)

        while len(buffer) - start >= tokens_per_batch:
            batch_tokens = buffer[
                start:start + tokens_per_batch
            ]

            start += tokens_per_batch

            batch = torch.tensor(
                batch_tokens,
                dtype=torch.long,
            ).reshape(batch_size, block_size)

            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            yield inputs, targets

        # Periodically remove the consumed prefix.
        if start >= 10 * tokens_per_batch:
            buffer = buffer[start:]
            start = 0




'''This function performs training'''

def logit_loss(logitx: torch.Tensor, y : torch.Tensor):
    # logitx has dimensions (batch_size, ..., seq, vocab_size)
    # y is the input of token ids of dimensions (batch_size, ..., seq)
    
    targets = y # with dimensions (batch_size, ..., seq -1) #corresponds to inserting 'target' of the batch_stream
    logits = logitx # with dimensions (batch_size, ..., seq-1, vocab_size). Corresponds to inserting 'input' of the batch_stream
    log_probs = torch.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(
        -1,
        targets.unsqueeze(-1)
    ).squeeze(-1)
   

    return -target_log_probs.mean()



def trainingSmallLM(AdamWOptx: torch.optim.Optimizer, transnewLM: nn.Module, batch_stream: BatchStream, iter_num: int, iter_per_epoch:int, eta_min: float, eta_max: float, T_w: int, n_check: int, output_file: str, init_loss: float, learning_schedule: bool, loss_history: list, init_learn:int, coup: float, device = None):
    # n_text_range is the range of docs
    # n_chunk is the size of each chunk
    # init_learn is the point in the learning schedule i will start from 
    #enc is the encoder
    checkpoint = {
                    "model": copy.deepcopy(transnewLM.state_dict()),
                    "optimizer": copy.deepcopy(AdamWOptx.state_dict()),
                    "loss_best": init_loss,
                    "loss_history": list(loss_history),
                    "iter_num": 0
                }
    loss_average = [] #this is the normal global loss
    loss_phrase_average = []
    loss_total_average = []
    loss_best = init_loss
    new_average = init_loss
    input, target = next(batch_stream)
    if input is None:
        return None
    input = input.to(device)
    target = target.to(device)
    
    positions0 = torch.tensor(range(input.shape[-1])) 
    positions = positions0.expand_as(input).to(device)
    eta_i = None
    checkpoint = None
    for i in range(iter_num):
        if learning_schedule == True:
            eta_i = eta_schedule(eta_max, eta_min, T_w, iter_per_epoch, 0.3, 1.5, (i+init_learn)%iter_per_epoch )
            for group in AdamWOptx.param_groups:
                group["alpha"] = eta_i
        # print("eta_i = ", eta_i)
        AdamWOptx.zero_grad()
        logits_global = transnewLM(input, positions, coup)
        loss1 = logit_loss(logits_global, target)
        loss_history.append(loss1.item())
        loss_average.append(loss1.item())
        loss1.backward()
        AdamWOptx.step()
        if (i+1)%n_check ==0:
            new_average = float(np.mean(loss_average)) #global
            print("num_iter = ", init_learn + i,", current loss = ",  new_average , ' loss_best = ', loss_best,  ', learning rate = ', eta_i)
            loss_average = []
            if new_average < loss_best:
                loss_best = new_average
                #print("saving")
                checkpoint = {
                    "model": copy.deepcopy(transnewLM.state_dict()),
                    "optimizer": copy.deepcopy(AdamWOptx.state_dict()),
                    "loss_best": loss_best,
                    "loss_history": list(loss_history),
                    "iter_num": i
                }
                # torch.save(checkpoint, output_file+'.pt')
        if i < iter_num - 1:
            try:
                input, target = next(batch_stream) 
            except StopIteration:
                print("the batch stream ended")
                print("saving")
                torch.save({"model": copy.deepcopy(transnewLM.state_dict()), "optimizer": copy.deepcopy(AdamWOptx.state_dict()), "loss_history": list(loss_history)},  output_file)
                break        
            input = input.to(device)
            target = target.to(device)
        if (init_learn + i)%iter_per_epoch == 0:
            if checkpoint is not None and device == torch.device("cuda"):
                torch.save({"model": copy.deepcopy(transnewLM.state_dict()), "optimizer": copy.deepcopy(AdamWOptx.state_dict()), "loss_history": list(loss_history)},  output_file)
    # torch.save(checkpoint, output_file+'.pt')
    return None


def trainingStandardLM(AdamWOptx: torch.optim.Optimizer, transnewLM: nn.Module, batch_stream: BatchStream, iter_num: int, iter_per_epoch:int, eta_min: float, eta_max: float, T_w: int, n_check: int, output_file: str, init_loss: float, learning_schedule: bool, loss_history: list, init_learn:int, device = None):
    # n_text_range is the range of docs
    # n_chunk is the size of each chunk
    # init_learn is the point in the learning schedule i will start from 
    #enc is the encoder
    checkpoint = {
                    "model": copy.deepcopy(transnewLM.state_dict()),
                    "optimizer": copy.deepcopy(AdamWOptx.state_dict()),
                    "loss_best": init_loss,
                    "loss_history": list(loss_history),
                    "iter_num": 0
                }
    loss_average = [] #this is the normal global loss
    loss_phrase_average = []
    loss_total_average = []
    loss_best = init_loss
    new_average = init_loss
    input, target = next(batch_stream)
    if input is None:
        return None
    input = input.to(device)
    target = target.to(device)
    
    positions0 = torch.tensor(range(input.shape[-1])) 
    positions = positions0.expand_as(input).to(device)
    eta_i = None
    checkpoint = None
    for i in range(iter_num):
        if learning_schedule == True:
            eta_i = eta_schedule(eta_max, eta_min, T_w, iter_per_epoch, 0.3, 1.5, (i+init_learn)%iter_per_epoch )
            for group in AdamWOptx.param_groups:
                group["alpha"] = eta_i
        # print("eta_i = ", eta_i)
        AdamWOptx.zero_grad()
        logits_global = transnewLM(input, positions)
        loss1 = logit_loss(logits_global, target)
        loss_history.append(loss1.item())
        loss_average.append(loss1.item())
        loss1.backward()
        AdamWOptx.step()
        if (i+1)%n_check ==0:
            new_average = float(np.mean(loss_average)) #global
            print("num_iter = ", init_learn + i,", current loss = ",  new_average , ' loss_best = ', loss_best,  ', learning rate = ', eta_i)
            loss_average = []
            if new_average < loss_best:
                loss_best = new_average
                # print("saving")
                # torch.save({"model": copy.deepcopy(transnewLM.state_dict()), "optimizer": copy.deepcopy(AdamWOptx.state_dict()), "loss_history": list(loss_history)}, output_file)
        if i < iter_num - 1:
            try:
                input, target = next(batch_stream) 
            except StopIteration:
                print("the batch stream ended")
                print("saving")
                checkpoint = {
                    "model": copy.deepcopy(transnewLM.state_dict()),
                    "optimizer": copy.deepcopy(AdamWOptx.state_dict()),
                    "loss_best": loss_best,
                    "loss_history": list(loss_history),
                    "iter_num": i
                }
                torch.save({"model": copy.deepcopy(transnewLM.state_dict()), "optimizer": copy.deepcopy(AdamWOptx.state_dict()), "loss_history": list(loss_history)}, output_file)
                break
            input = input.to(device)
            target = target.to(device)
        if (init_learn + i)%iter_per_epoch == 0:
            if checkpoint is not None and device == torch.device("cuda"):
                torch.save({"model": copy.deepcopy(transnewLM.state_dict()), "optimizer": copy.deepcopy(AdamWOptx.state_dict()), "loss_history": list(loss_history)}, output_file)
    # torch.save(checkpoint, output_file+'.pt')
    return None    

def import_model(model, path, device):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only= False,
    )
    model.load_state_dict(checkpoint["model"])
    return None

def validation_model(model: nn.Module, batch_stream: BatchStream, coup: float, device, iteration_model: int):
    model.eval()
    validation_loss = []
    positions = None
    with torch.inference_mode():
        for i in range(iteration_model):
            if i % 50 == 0:
                print(f"\rinteration number  {i}", end="", flush=True)
            inputs, target = next(batch_stream)
            inputs = inputs.to(device)
            target = target.to(device)
            if positions is None:
                positions0 = torch.tensor(range(inputs.shape[-1])) 
                positions = positions0.expand_as(inputs).to(device)
            logits = model(inputs, positions, coup)
            loss1 = logit_loss(logits, target)
            validation_loss.append(loss1.item())
    
    return validation_loss
    
def validation_losses(coup, model, model_paths, iter_skip, iter_valid, device, get_data_stream, enc, max_seq_len, batch_size, docs_per_tokenize):
    results = {}
    ind = 1
    for x in model_paths:
        print("starting new model parameters")
        import_model(model, x, device)
        train_iter = get_data_stream(restart = True)
        batch_stream = token_batch_stream(
            train_iter,
            enc,
            max_seq_len,
            batch_size,
            docs_per_tokenize
        )
        for i in range(iter_skip):
            next(batch_stream)
        results[ind] = validation_model(model, batch_stream, coup, device, iter_valid)
        ind = ind + 1
    return results


@torch.inference_mode()
def decodeNewModel(coup : float, TransnewLM: nn.Module, tokens: torch.Tensor, token_positions: torch.Tensor, output_length: int):
    # TransLM is the model
    # tokens is the tensor containing tokens with dimensions (batch_size, ..., seq)
    # token_positions is the tensor containing positions with dimensions (batch_size, ..., seq)
    for i in range(output_length):
        logitsx = TransnewLM(tokens, token_positions, coup)
        logitsx_last  = logitsx[..., -1, :]
        logitsx_last_softmax = soft_max(logitsx_last, dimen = -1)
        logitsx_last_softmax_flat = logitsx_last_softmax.reshape(-1, logitsx_last_softmax.shape[-1])
        samples = torch.multinomial(logitsx_last_softmax_flat, 1)
        samples= samples.reshape(logitsx_last_softmax.shape[:-1]) #this has dimensions (batch_size, ...)
        tokens = torch.cat([tokens, samples.unsqueeze(-1)], dim = -1)
        new_pos = token_positions[..., -1:] + 1
        token_positions = torch.cat([token_positions, new_pos], dim = -1)
    return tokens


@torch.inference_mode()
def decodeModel(TransLM: nn.Module, tokens: torch.Tensor, token_positions: torch.Tensor, output_length: int):
    # TransLM is the model
    # tokens is the tensor containing tokens with dimensions (batch_size, ..., seq)
    # token_positions is the tensor containing positions with dimensions (batch_size, ..., seq)
    for i in range(output_length):
        logitsx = TransLM(tokens, token_positions)
        logitsx_last  = logitsx[..., -1, :]
        logitsx_last_softmax = soft_max(logitsx_last, dimen = -1)
        logitsx_last_softmax_flat = logitsx_last_softmax.reshape(-1, logitsx_last_softmax.shape[-1])
        samples = torch.multinomial(logitsx_last_softmax_flat, 1)
        samples= samples.reshape(logitsx_last_softmax.shape[:-1]) #this has dimensions (batch_size, ...)
        tokens = torch.cat([tokens, samples.unsqueeze(-1)], dim = -1)
        new_pos = token_positions[..., -1:] + 1
        token_positions = torch.cat([token_positions, new_pos], dim = -1)
    return tokens

def decode_new(tokenizer, text: str, coup:float, model: nn.Module, token_length:int, device):
    tokensRaw = tokenizer.encode(text)
    tokensRawlen = len(tokensRaw)
    tokensA = torch.tensor(tokensRaw).reshape(1, tokensRawlen).to(device)
    positionsA = torch.tensor(range(tokensRawlen)).reshape(1, tokensRawlen).to(device)
    out = decodeNewModel(coup, model, tokensA, positionsA, token_length)[0].tolist()
    return tokenizer.decode(out)

def decode_old(tokenizer, text: str, model: nn.Module, token_length:int, device):
    tokensRaw = tokenizer.encode(text)
    tokensRawlen = len(tokensRaw)
    tokensA = torch.tensor(tokensRaw).reshape(1, tokensRawlen).to(device)
    positionsA = torch.tensor(range(tokensRawlen)).reshape(1, tokensRawlen).to(device)
    out = decodeModel(model, tokensA, positionsA, token_length)[0].tolist()
    return tokenizer.decode(out)   
  


def import_phrase_weights(
    combined_model,
    phrase_checkpoint_path,
    device,
    freeze_phrase=False,
    trusted_checkpoint=True,
):
    """
    Imports a pretrained PhraseTransformerLM into the phrase
    components of NewTransformerLM.

    Expected source prefixes:
        Embedding.*
        TransBlockList.*

    Destination prefixes:
        Embedding_phrase.*
        PhraseTransBlockList.*
    """

    checkpoint = torch.load(
        phrase_checkpoint_path,
        map_location=device,
        weights_only=not trusted_checkpoint,
    )

    # Support either:
    #   torch.save(model.state_dict(), path)
    # or:
    #   torch.save({"model": model.state_dict(), ...}, path)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        source_state = checkpoint["model"]
    else:
        source_state = checkpoint

    phrase_state = {}

    for original_key, value in source_state.items():
        # Handle checkpoints saved using DataParallel.
        key = original_key

        if key.startswith("module."):
            key = key[len("module."):]

        if key.startswith("Embedding."):
            destination_key = key.replace(
                "Embedding.",
                "Embedding_phrase.",
                1,
            )
            phrase_state[destination_key] = value

        elif key.startswith("TransBlockList."):
            destination_key = key.replace(
                "TransBlockList.",
                "PhraseTransBlockList.",
                1,
            )
            phrase_state[destination_key] = value

    if not phrase_state:
        raise ValueError(
            "No phrase weights were found. Expected checkpoint "
            "keys beginning with 'Embedding.' or 'TransBlockList.'."
        )

    load_result = combined_model.load_state_dict(
        phrase_state,
        strict=False,
    )

    # Unexpected keys indicate that the renamed phrase keys do not
    # correspond to components in the combined model.
    if load_result.unexpected_keys:
        raise ValueError(
            "Unexpected phrase keys after renaming:\n"
            + "\n".join(load_result.unexpected_keys)
        )

    if freeze_phrase:
        for parameter in (
            combined_model.Embedding_phrase.parameters()
        ):
            parameter.requires_grad = False

        for block in combined_model.PhraseTransBlockList:
            for parameter in block.parameters():
                parameter.requires_grad = False

    print(
        f"Imported {len(phrase_state)} phrase tensors."
    )

    if freeze_phrase:
        print("The helper backbone has been frozen.")
    else:
        print("The helper backbone remains trainable.")

    return combined_model

