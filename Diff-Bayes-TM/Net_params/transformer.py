import math
import torch
import torch.nn.functional as F

from torch import nn
from Net_params.useful_modules import AdaLayerNorm, GELU2
from Net_params.useful_modules import Conv_MLP, LearnablePositionalEncoding


class FullAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1
    ):
        super().__init__()
        assert n_embd % n_head == 0

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)


        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        att = att.mean(dim=1, keepdim=False)


        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        condition_embd: int,
        n_head: int,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
    ):
        super().__init__()
        assert n_embd % n_head == 0

        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()

        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        att = att.mean(dim=1, keepdim=False)


        y = self.resid_drop(self.proj(y))
        return y, att
    

class EncoderBlock(nn.Module):

    def __init__(
        self,
        n_embd: int = 1024,
        n_head: int = 16,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_hidden_times: int = 4,
        activate: str ='GELU'
    ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
            )
        
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )
        
    def forward(self, x, timestep, mask=None, label_emb=None):
        a, att = self.attn(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, att


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer: int = 14,
        n_embd: int = 1024,
        n_head: int = 16,
        attn_pdrop: float = 0.,
        resid_pdrop: float = 0.,
        mlp_hidden_times: int = 4,
        block_activate: str = 'GELU',
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[EncoderBlock(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
        ) for _ in range(n_layer)])

    def forward(self, input, t, padding_masks=None, label_emb=None, return_att=False):
        x = input
        att_weights = []
        for block_idx in range(len(self.blocks)):
            x, att_weight = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)
            att_weights.append(att_weight)
        
        if return_att:
            return x, att_weights
        return x


class DecoderBlock(nn.Module):

    def __init__(
        self,
        n_embd: int = 1024,
        n_head: int = 16,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_hidden_times: int = 4,
        activate: str = 'GELU',
        condition_dim: int = 1024
    ):
        super().__init__()
        
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn1 = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop,
                )
        self.attn2 = CrossAttention(
                n_embd=n_embd,
                condition_embd=condition_dim,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                )
        
        self.ln1_1 = AdaLayerNorm(n_embd)

        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, encoder_output, timestep, mask=None, label_emb=None):
        a, att = self.attn1(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        a, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
    

class Decoder(nn.Module):
    def __init__(
        self,
        n_embd: int = 1024,
        n_head: int = 16,
        n_layer: int = 10,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_hidden_times: int = 4,
        block_activate: str = 'GELU',
        condition_dim: int = 512 
    ):
      super().__init__()
      self.d_model = n_embd
      self.blocks = nn.Sequential(*[DecoderBlock(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                condition_dim=condition_dim,
        ) for _ in range(n_layer)])
      
    def forward(self, x, t, enc, padding_masks=None, label_emb=None, return_att=False):
        for block_idx in range(len(self.blocks)):
            x = self.blocks[block_idx](x, enc, t, mask=padding_masks, label_emb=label_emb)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        n_feat: int = 144,
        n_layer_enc: int = 5,
        n_layer_dec: int = 14,
        n_embd: int = 1024,
        n_heads: int = 16,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_hidden_times: int = 2,
        block_activate='GELU',
        max_len: int = 2048
    ):
        super().__init__()
        self.feature_size = n_feat
        self.emb = Conv_MLP(n_feat, n_embd)
        self.inverse = Conv_MLP(n_embd, n_feat)

        self.encoder = Encoder(n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate)
        self.pos_enc = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

        self.decoder = Decoder(n_embd, n_heads, n_layer_dec, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate, condition_dim=n_embd)
        self.pos_dec = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)
        self.act = nn.Sigmoid()
        self.log_scale = nn.Parameter(torch.zeros(1, 1, n_feat))

    @property
    def scale(self):
        return self.act(self.log_scale)

    def forward(self, input, t, padding_masks=None):
        b, c, h = input.shape
        emb = self.emb(input)
        inp_enc = self.pos_enc(emb)
        enc_cond = self.encoder(inp_enc, t, padding_masks=padding_masks)

        inp_dec = self.pos_dec(emb)
        x = self.decoder(inp_dec, t, enc_cond, padding_masks=padding_masks)
        output = self.act(self.inverse(x)) * self.scale
        return output
