import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn.modules.transformer import _get_clones
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from torch.nn.init import *

from torch.nn.modules.activation import MultiheadAttention
from torch.nn.functional import linear, softmax, dropout
from utilities.device import get_device
from utilities.argument_funcs import parse_train_args

# TransformerEncoderRPR
class TransformerEncoderRPR(Module):

    def __init__(self, encoder_layer, num_layers, encoder_past, max_seq, c_max_seq, b_max_seq, norm=None):
        super(TransformerEncoderRPR, self).__init__()
        self.past_layers = _get_clones(encoder_past, 1)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.max_seq = max_seq
        self.c_max_seq = c_max_seq
        self.b_max_seq = b_max_seq

    def forward(self, src, mask=None, src_key_padding_mask=None):

        args = parse_train_args()

        def generate_square_subsequent_mask(sz: int) -> Tensor:
            r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
                Unmasked positions are filled with float(0.0).
            """
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask

        x_n = src[:mask.shape[0], :, :]
        x_c = src[mask.shape[0]:mask.shape[0]+(src.shape[0] // 10 + 1), :, :]
        x_b = src[mask.shape[0]+(src.shape[0] // 10 + 1):, :, :]

        if args.gpu[0] != -1:
            mask_c = generate_square_subsequent_mask(x_c.shape[0]).cuda(device=args.gpu[0])
            mask_b = generate_square_subsequent_mask(x_b.shape[0]).cuda(device=args.gpu[0])
            mask_zero_c = torch.zeros(x_n.shape[0], x_c.shape[0]).cuda(device=args.gpu[0])
            mask_zero_b = torch.zeros(x_n.shape[0], x_b.shape[0]).cuda(device=args.gpu[0])
        else:
            mask_c = generate_square_subsequent_mask(x_c.shape[0]).cpu()
            mask_b = generate_square_subsequent_mask(x_b.shape[0]).cpu()
            mask_zero_c = torch.zeros(x_n.shape[0], x_c.shape[0]).cpu()
            mask_zero_b = torch.zeros(x_n.shape[0], x_b.shape[0]).cpu()

        mask_past_layer = torch.cat((mask, mask_zero_c, mask_zero_b), dim=1)

        # past layer of transformer
        output = self.past_layers[0](x_n, x_c, x_b, src_past_c_mask=mask_c, src_past_c_key_padding_mask=src_key_padding_mask,
                                     src_past_b_mask=mask_b, src_past_b_key_padding_mask=src_key_padding_mask,
                                     src_mask=mask_past_layer, src_key_padding_mask=src_key_padding_mask)

        # origin Transformer
        for i in range(1, self.num_layers):
            output = self.layers[i](output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output


class TransformerEncoderLayerRPR_(Module):
    """
    ----------
    The model of condition input
    ----------
    """

    def __init__(self, cpast_layer_dmodel, cpast_layer_nhead, cpast_dim_forward,
                 bpast_layer_dmodel, bpast_layer_nhead, bpast_dim_forward,
                 d_model, nhead, dim_feedforward=2048, dropout=0.1, er_len=None):
        super(TransformerEncoderLayerRPR_, self).__init__()

        # past_layer of chord
        self.C_self_attn1 = MultiheadAttentionRPR(cpast_layer_dmodel, cpast_layer_nhead, dropout=dropout, er_len=er_len)
        # Implementation of Feedforward model
        self.C_linear1 = Linear(cpast_layer_dmodel, cpast_dim_forward)
        self.C_dropout = Dropout(dropout)
        self.C_linear2 = Linear(cpast_dim_forward, cpast_layer_dmodel)

        self.C_norm1 = LayerNorm(cpast_layer_dmodel)
        self.C_norm2 = LayerNorm(cpast_layer_dmodel)
        self.C_dropout1 = Dropout(dropout)
        self.C_dropout2 = Dropout(dropout)

        # Implementation of linear for kc and vc
        self.C_norm3 = LayerNorm(cpast_layer_dmodel)
        self.C_norm4 = LayerNorm(cpast_layer_dmodel)
        self.C_dropout3 = Dropout(dropout)
        self.C_dropout4 = Dropout(dropout)
        self.C_linear3 = Linear(cpast_layer_dmodel, cpast_layer_dmodel)
        self.C_linear4 = Linear(cpast_layer_dmodel, cpast_layer_dmodel)

        # past_layer of beat
        self.B_self_attn1 = MultiheadAttentionRPR(bpast_layer_dmodel, bpast_layer_nhead, dropout=dropout, er_len=er_len)
        # Implementation of Feedforward model
        self.B_linear1 = Linear(bpast_layer_dmodel, bpast_dim_forward)
        self.B_dropout = Dropout(dropout)
        self.B_linear2 = Linear(bpast_dim_forward, bpast_layer_dmodel)

        self.B_norm1 = LayerNorm(bpast_layer_dmodel)
        self.B_norm2 = LayerNorm(bpast_layer_dmodel)
        self.B_dropout1 = Dropout(dropout)
        self.B_dropout2 = Dropout(dropout)

        # Implementation of linear for kc and vc
        self.B_norm3 = LayerNorm(bpast_layer_dmodel)
        self.B_norm4 = LayerNorm(bpast_layer_dmodel)
        self.B_dropout3 = Dropout(dropout)
        self.B_dropout4 = Dropout(dropout)
        self.B_linear3 = Linear(bpast_layer_dmodel, bpast_layer_dmodel)
        self.B_linear4 = Linear(bpast_layer_dmodel, bpast_layer_dmodel)

        # normal encoder
        self.self_attn2 = MultiheadAttentionRPR(d_model, nhead, dropout=dropout)
        self.linear5 = Linear(d_model, dim_feedforward)
        self.dropout5 = Dropout(dropout)
        self.linear6 = Linear(dim_feedforward, d_model)

        self.norm5 = LayerNorm(d_model)
        self.norm6 = LayerNorm(d_model)
        self.dropout6 = Dropout(dropout)
        self.dropout7 = Dropout(dropout)

    def forward(self, src_n_past, src_c_past, src_b_past, src_past_c_mask=None, src_past_c_key_padding_mask=None,
                src_past_b_mask=None, src_past_b_key_padding_mask=None,
                src_mask=None, src_key_padding_mask=None):
        # past layer of chord:
        # calculate k_c,v_c first:
        src_C = src_c_past
        src_C_past2 = self.C_self_attn1(src_C, src_C, src_C, attn_mask=src_past_c_mask,
                                        key_padding_mask=src_past_c_key_padding_mask)[0]
        src_C = src_C + self.C_dropout1(src_C_past2)
        src_C = self.C_norm1(src_C)
        src_C_past2 = self.C_linear2(self.C_dropout(F.relu(self.C_linear1(src_C))))
        src_C = src_C + self.C_dropout2(src_C_past2)
        src_C = self.C_norm2(src_C)

        kc = self.C_norm3(self.C_dropout3(self.C_linear3(src_C)))
        vc = self.C_norm4(self.C_dropout4(self.C_linear4(src_C)))

        # calculate Beat_k and Beat_v:
        src_b = src_b_past
        src_b_past2 = self.B_self_attn1(src_b, src_b, src_b, attn_mask=src_past_b_mask,
                                        key_padding_mask=src_past_b_key_padding_mask)[0]
        src_b = src_b + self.B_dropout1(src_b_past2)
        src_b = self.B_norm1(src_b)
        src_b_past2 = self.B_linear2(self.B_dropout(F.relu(self.B_linear1(src_b))))
        src_b = src_b + self.B_dropout2(src_b_past2)
        src_b = self.C_norm2(src_b)

        kb = self.B_norm3(self.B_dropout3(self.B_linear3(src_b)))
        vb = self.B_norm4(self.B_dropout4(self.B_linear4(src_b)))

        # # layer0:
        k = torch.cat((src_n_past, kc, kb), dim=0)
        v = torch.cat((src_n_past, vc, vb), dim=0)
        src2 = self.self_attn2(src_n_past, k, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        # src2 = self.self_attn2(src, src, src, attn_mask=src_mask[:, :2048], key_padding_mask=src_key_padding_mask)[0]
        src = src_n_past + self.dropout5(src2)
        src = self.norm5(src)
        src2 = self.linear6(self.dropout6(F.relu(self.linear5(src2))))
        src = src + self.dropout7(src2)
        src = self.norm6(src)
        return src


# TransformerEncoderLayerRPR
class TransformerEncoderLayerRPR(Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, er_len=None):
        super(TransformerEncoderLayerRPR, self).__init__()
        self.self_attn = MultiheadAttentionRPR(d_model, nhead, dropout=dropout, er_len=er_len)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src: Tensor, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerEncoderPastLayer(Module):

    def __init__(self, cpast_layer_dmodel, cpast_layer_nhead, cpast_dim_forward,
                 bpast_layer_dmodel, bpast_layer_nhead, bpast_dim_forward,
                 d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderPastLayer, self).__init__()

        # past_layer of chord
        self.C_self_attn1 = MultiheadAttention(cpast_layer_dmodel, cpast_layer_nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.C_linear1 = Linear(cpast_layer_dmodel, cpast_dim_forward)
        self.C_dropout = Dropout(dropout)
        self.C_linear2 = Linear(cpast_dim_forward, cpast_layer_dmodel)

        self.C_norm1 = LayerNorm(cpast_layer_dmodel)
        self.C_norm2 = LayerNorm(cpast_layer_dmodel)
        self.C_dropout1 = Dropout(dropout)
        self.C_dropout2 = Dropout(dropout)

        # Implementation of linear for kc and vc
        self.C_norm3 = LayerNorm(cpast_layer_dmodel)
        self.C_norm4 = LayerNorm(cpast_layer_dmodel)
        self.C_dropout3 = Dropout(dropout)
        self.C_dropout4 = Dropout(dropout)
        self.C_linear3 = Linear(cpast_layer_dmodel, cpast_layer_dmodel)
        self.C_linear4 = Linear(cpast_layer_dmodel, cpast_layer_dmodel)

        # past_layer of beat
        self.B_self_attn1 = MultiheadAttention(bpast_layer_dmodel, bpast_layer_nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.B_linear1 = Linear(bpast_layer_dmodel, bpast_dim_forward)
        self.B_dropout = Dropout(dropout)
        self.B_linear2 = Linear(bpast_dim_forward, bpast_layer_dmodel)

        self.B_norm1 = LayerNorm(bpast_layer_dmodel)
        self.B_norm2 = LayerNorm(bpast_layer_dmodel)
        self.B_dropout1 = Dropout(dropout)
        self.B_dropout2 = Dropout(dropout)

        # Implementation of linear for kc and vc
        self.B_norm3 = LayerNorm(bpast_layer_dmodel)
        self.B_norm4 = LayerNorm(bpast_layer_dmodel)
        self.B_dropout3 = Dropout(dropout)
        self.B_dropout4 = Dropout(dropout)
        self.B_linear3 = Linear(bpast_layer_dmodel, bpast_layer_dmodel)
        self.B_linear4 = Linear(bpast_layer_dmodel, bpast_layer_dmodel)

        # normal encoder
        self.self_attn2 = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear5 = Linear(d_model, dim_feedforward)
        self.dropout5 = Dropout(dropout)
        self.linear6 = Linear(dim_feedforward, d_model)

        self.norm5 = LayerNorm(d_model)
        self.norm6 = LayerNorm(d_model)
        self.dropout6 = Dropout(dropout)
        self.dropout7 = Dropout(dropout)

    def forward(self, src_n_past, src_c_past, src_b_past, src_past_c_mask=None, src_past_c_key_padding_mask=None,
                src_past_b_mask=None, src_past_b_key_padding_mask=None,
                src_mask=None, src_key_padding_mask=None):
        # past layer of chord:
        # calculate k_c,v_c first:
        src_C = src_c_past
        src_C_past2 = self.C_self_attn1(src_C, src_C, src_C, attn_mask=src_past_c_mask,
                                        key_padding_mask=src_past_c_key_padding_mask)[0]
        src_C = src_C + self.C_dropout1(src_C_past2)
        src_C = self.C_norm1(src_C)
        src_C_past2 = self.C_linear2(self.C_dropout(F.relu(self.C_linear1(src_C))))
        src_C = src_C + self.C_dropout2(src_C_past2)
        src_C = self.C_norm2(src_C)

        kc = self.C_norm3(self.C_dropout3(self.C_linear3(src_C)))
        vc = self.C_norm4(self.C_dropout4(self.C_linear4(src_C)))

        # calculate Beat_k and Beat_v:
        src_b = src_b_past
        src_b_past2 = self.B_self_attn1(src_b, src_b, src_b, attn_mask=src_past_b_mask,
                                        key_padding_mask=src_past_b_key_padding_mask)[0]
        src_b = src_b + self.B_dropout1(src_b_past2)
        src_b = self.B_norm1(src_b)
        src_b_past2 = self.B_linear2(self.B_dropout(F.relu(self.B_linear1(src_b))))
        src_b = src_b + self.B_dropout2(src_b_past2)
        src_b = self.C_norm2(src_b)

        kb = self.B_norm3(self.B_dropout3(self.B_linear3(src_b)))
        vb = self.B_norm4(self.B_dropout4(self.B_linear4(src_b)))

        # # layer0:
        k = torch.cat((src_n_past, kc, kb), dim=0)
        v = torch.cat((src_n_past, vc, vb), dim=0)
        src2 = self.self_attn2(src_n_past, k, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        # src2 = self.self_attn2(src, src, src, attn_mask=src_mask[:, :2048], key_padding_mask=src_key_padding_mask)[0]
        src = src_n_past + self.dropout5(src2)
        src = self.norm5(src)
        src2 = self.linear6(self.dropout6(F.relu(self.linear5(src2))))
        src = src + self.dropout7(src2)
        src = self.norm6(src)
        return src


class TransformerEncoderLayer(Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src: Tensor, src_mask=None, src_key_padding_mask=None):
        # if src.size()[0] == src_mask.size()[0]:
        #     src2 = self.self_attn(src, src, src, attn_mask=src_mask,
        #                           key_padding_mask=src_key_padding_mask)[0]
        # else:
        #     src2 = src[:2048,:,:]
        #     key = src[:2304,:,:]
        #     value = torch.cat((src2, src[2304:2560,:,:]), dim=0)
        #     src = src2
        #     src2 = self.self_attn_2(src2,key,value,attn_mask=src_mask,
        #                           key_padding_mask=src_key_padding_mask)[0]

        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerEncoder(Module):


    def __init__(self, encoder_layer, num_layers, encoder_past, max_seq, c_max_seq, b_max_seq, norm=None):
        super(TransformerEncoder, self).__init__()
        self.past_layers = _get_clones(encoder_past, 1)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.max_seq = max_seq
        self.c_max_seq = c_max_seq
        self.b_max_seq = b_max_seq

    def forward(self, src, mask=None, src_key_padding_mask=None):

        args = parse_train_args()

        def generate_square_subsequent_mask(sz: int) -> Tensor:
            r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
                Unmasked positions are filled with float(0.0).
            """
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask


        x_n = src[:mask.shape[0], :, :]
        x_c = src[mask.shape[0]:mask.shape[0]+(src.shape[0] // 10 + 1), :, :]
        x_b = src[mask.shape[0]+(src.shape[0] // 10 + 1):, :, :]


        if args.gpu[0] != -1:
            mask_c = generate_square_subsequent_mask(x_c.shape[0]).cuda(device=args.gpu[0])
            mask_b = generate_square_subsequent_mask(x_b.shape[0]).cuda(device=args.gpu[0])
            mask_zero_c = torch.zeros(x_n.shape[0], x_c.shape[0]).cuda(device=args.gpu[0])
            mask_zero_b = torch.zeros(x_n.shape[0], x_b.shape[0]).cuda(device=args.gpu[0])
        else:
            mask_c = generate_square_subsequent_mask(x_c.shape[0]).cpu()
            mask_b = generate_square_subsequent_mask(x_b.shape[0]).cpu()
            mask_zero_c = torch.zeros(x_n.shape[0], x_c.shape[0]).cpu()
            mask_zero_b = torch.zeros(x_n.shape[0], x_b.shape[0]).cpu()

        mask_past_layer = torch.cat((mask, mask_zero_c, mask_zero_b), dim=1)

        # past layer of transformer
        output = self.past_layers[0](x_n, x_c, x_b, src_past_c_mask=mask_c, src_past_c_key_padding_mask=src_key_padding_mask,
                                     src_past_b_mask=mask_b, src_past_b_key_padding_mask=src_key_padding_mask,
                                     src_mask=mask_past_layer, src_key_padding_mask=src_key_padding_mask)

        # origin Transformer
        for i in range(1, self.num_layers):
            output = self.layers[i](output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        # x2 = src[self.max_seq:, :, :]
        # mask2 = generate_square_subsequent_mask(x2.shape[0]).to(get_device())
        # out_past = self.past_layers(output, src_mask=mask2, src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output



# MultiheadAttentionRPR
class MultiheadAttentionRPR(Module):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None, er_len=None):
        super(MultiheadAttentionRPR, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        # Adding RPR embedding matrix
        if (er_len is not None):
            self.Er = Parameter(torch.rand((er_len, self.head_dim), dtype=torch.float32))
        else:
            self.Er = None

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):

        if hasattr(self, '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
            # return F.multi_head_attention_forward(
            #     query, key, value, self.embed_dim, self.num_heads,
            #     self.in_proj_weight, self.in_proj_bias,
            #     self.bias_k, self.bias_v, self.add_zero_attn,
            #     self.dropout, self.out_proj.weight, self.out_proj.bias,
            #     training=self.training,
            #     key_padding_mask=key_padding_mask, need_weights=need_weights,
            #     attn_mask=attn_mask, use_separate_proj_weight=True,
            #     q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
            #     v_proj_weight=self.v_proj_weight)

            return multi_head_attention_forward_rpr(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, rpr_mat=self.Er)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttention module has been implemented. \
                    Please re-train your model with the new module',
                              UserWarning)

            # return F.multi_head_attention_forward(
            #     query, key, value, self.embed_dim, self.num_heads,
            #     self.in_proj_weight, self.in_proj_bias,
            #     self.bias_k, self.bias_v, self.add_zero_attn,
            #     self.dropout, self.out_proj.weight, self.out_proj.bias,
            #     training=self.training,
            #     key_padding_mask=key_padding_mask, need_weights=need_weights,
            #     attn_mask=attn_mask)

            return multi_head_attention_forward_rpr(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, rpr_mat=self.Er)

# multi_head_attention_forward_rpr
def multi_head_attention_forward_rpr(query,  # type: Tensor
                                     key,  # type: Tensor
                                     value,  # type: Tensor
                                     embed_dim_to_check,  # type: int
                                     num_heads,  # type: int
                                     in_proj_weight,  # type: Tensor
                                     in_proj_bias,  # type: Tensor
                                     bias_k,  # type: Optional[Tensor]
                                     bias_v,  # type: Optional[Tensor]
                                     add_zero_attn,  # type: bool
                                     dropout_p,  # type: float
                                     out_proj_weight,  # type: Tensor
                                     out_proj_bias,  # type: Tensor
                                     training=True,  # type: bool
                                     key_padding_mask=None,  # type: Optional[Tensor]
                                     need_weights=True,  # type: bool
                                     attn_mask=None,  # type: Optional[Tensor]
                                     use_separate_proj_weight=False,  # type: bool
                                     q_proj_weight=None,  # type: Optional[Tensor]
                                     k_proj_weight=None,  # type: Optional[Tensor]
                                     v_proj_weight=None,  # type: Optional[Tensor]
                                     static_k=None,  # type: Optional[Tensor]
                                     static_v=None,  # type: Optional[Tensor]
                                     rpr_mat=None
                                     ):

    # type: (...) -> Tuple[Tensor, Optional[Tensor]]

    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if use_separate_proj_weight is not True:
        if qkv_same:
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif kv_same:
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                       torch.zeros((attn_mask.size(0), 1),
                                                   dtype=attn_mask.dtype,
                                                   device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                          dtype=attn_mask.dtype,
                                                          device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                               dtype=key_padding_mask.dtype,
                                               device=key_padding_mask.device)], dim=1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    ######### ADDITION OF RPR ###########
    if (rpr_mat is not None):
        rpr_mat = _get_valid_embedding(rpr_mat, q.shape[1], k.shape[1])
        qe = torch.einsum("hld,md->hlm", q, rpr_mat)
        srel = _skew(qe)

        attn_output_weights += srel

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = softmax(
        attn_output_weights, dim=-1)

    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


def _get_valid_embedding(Er, len_q, len_k):

    len_e = Er.shape[0]
    start = max(0, len_e - len_q)
    return Er[start:, :]


def _skew(qe):

    sz = qe.shape[1]
    mask = (torch.triu(torch.ones(sz, sz).to(qe.device)) == 1).float().flip(0)

    qe = mask * qe
    qe = F.pad(qe, (1, 0, 0, 0, 0, 0))
    qe = torch.reshape(qe, (qe.shape[0], qe.shape[2], qe.shape[1]))

    srel = qe[:, 1:, :]
    return srel


class Beat_Compute(Module):
    """
    This part is to calculate the k and v value of Beat
    """

    def __init__(self, past_layer_dmodel, past_layer_nhead, past_dim_forward,
                 d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(Beat_Compute, self).__init__()

        # Past layer for compute Beat
        self.self_attn1 = MultiheadAttention(past_layer_dmodel, past_layer_nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(past_layer_dmodel, past_dim_forward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(past_dim_forward, past_layer_dmodel)

        self.norm1 = LayerNorm(past_layer_dmodel)
        self.norm2 = LayerNorm(past_layer_dmodel)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Implementation of linear for Beat_k and Beat_v
        self.norm3 = LayerNorm(past_layer_dmodel)
        self.norm4 = LayerNorm(past_layer_dmodel)
        self.dropout3 = Dropout(dropout)
        self.dropout4 = Dropout(dropout)
        self.linear3 = Linear(past_layer_dmodel, past_layer_dmodel)
        self.linear4 = Linear(past_layer_dmodel, past_layer_dmodel)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        for i in range(4):
            src_temp = src[256 * i:256 * (i + 1), :, :]
            # calculate Beat_k and Beat_v:
            src_past2 = self.self_attn1(src_temp, src_temp, src_temp, attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)[0]
            src_past = src_temp + self.dropout1(src_past2)
            src_past = self.norm1(src_past)
            src_past2 = self.linear2(self.dropout(F.relu(self.linear1(src_past))))
            src_past = src_past + self.dropout2(src_past2)
            src_past = self.norm2(src_past)
            k = self.norm3(self.dropout3(self.linear3(src_past)))
            v = self.norm4(self.dropout4(self.linear4(src_past)))
            if i > 0:
                Beat_k = torch.cat((Beat_k, k), dim=0)
                Beat_v = torch.cat((Beat_v, v), dim=0)
            else:
                Beat_k = k
                Beat_v = v
        return Beat_k, Beat_v
