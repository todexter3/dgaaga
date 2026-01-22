import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding_group, FlattenHead
from layers.inventory_encoder import InventoryEncoder
from utils.tools import Transpose, MultiScaleConv

 

class Model(nn.Module):
    #一次性 encoder + 多尺度卷积聚合 
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.alpha = torch.nn.Parameter(torch.tensor(configs.tau_hat_init))
        padding = configs.stride
        patch_len = configs.patch_len
        stride = configs.stride
        self.configs = configs

        # feature groups
        self.feature_group = [[0],[1],[2,3,4,5,6],[7],[8]] # 1,1,5,1,1分组

        # 原始的 fea_fusion
        self.fea_fusion = nn.ModuleList()
        for i in range(len(self.feature_group)):
            layer = nn.Sequential(
                nn.Flatten(-2),
                nn.Linear(len(self.feature_group[i]) * configs.d_model, configs.d_model // 2),
                nn.Dropout(configs.dropout),
                nn.GELU()
            )
            self.fea_fusion.append(layer)

        self.patch_embedding = PatchEmbedding_group(configs.d_model, patch_len, stride, padding, configs.dropout, self.feature_group)

        # Encoder (共享 encoder)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False), 
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
                ) 
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        # 多尺度卷积
        self.multi_scale = MultiScaleConv(configs.d_model, kernel_sizes=(3,7,15))

        # Prediction Head 
        self.head_nf = int((configs.seq_len - patch_len) / stride + 2)
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(configs.dropout)
        self.feature_out = nn.Linear(configs.enc_in * configs.d_model, configs.d_model // 2)
        # projection 输入维度等于 len(feature_group) * configs.d_model * head_nf
        self.projection = nn.Linear(len(self.feature_group) * configs.d_model * self.head_nf, configs.output_channels)

  
    def regression(self, x_enc):
        # normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x / stdev

        B = x_enc.shape[0]
        x_in = x_enc.permute(0, 2, 1) 
        # patch embedding -> 得到每个 group 的 enc_out 列表
        enc_out_list, n_vars = self.patch_embedding(x_in)
        #一次性 concat 到 batch 维度并送入 encoder
        split_sizes = [t.shape[0] for t in enc_out_list]  
        enc_inputs = torch.cat(enc_out_list, dim=0)      
        # encoder
        enc_outputs, attns = self.encoder(enc_inputs)    
        # 卷积
        enc_outputs = self.multi_scale(enc_outputs)      

        # split 回各 group 并 reshape成 [B, n_vars, patch_num, d_model]
        splits = torch.split(enc_outputs, split_sizes, dim=0)  
        group_tensors = []
        for s in splits:
            patch_num = s.shape[1]
            d_model = s.shape[2]
            s = s.contiguous().view(B, 1, patch_num, d_model)
            s = s.permute(0, 1, 3, 2)
            group_tensors.append(s)

        # 在 d_model方向把不同 group 拼接
        out_cat = torch.cat(group_tensors, dim=2)

        # flatten across channel & patch and project
        output = self.flatten(out_cat)  
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  
        output = self.projection(output)  
        return output

    def forward(self, x_enc):   
        return self.regression(x_enc)
