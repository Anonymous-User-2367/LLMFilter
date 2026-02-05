import torch
import torch.nn as nn
from layers.mlp import MLP

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = int(configs.window_length / 2)
        self.enc_dim = self.token_len
        self.ed_ratio = int(configs.state_dim / configs.obs_dim)
        self.dec_dim = self.ed_ratio * self.token_len
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)
        self.hidden_dim_of_mlp = 512
        self.hidden_layers_of_mlp = 4
        self.mix = configs.mix_embeds
            
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))

        if configs.mlp_hidden_layers == 0:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.enc_dim, self.hidden_dim_of_mlp)
            self.decoder = nn.Linear(self.hidden_dim_of_mlp, self.dec_dim)
        else:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(self.enc_dim, self.hidden_dim_of_mlp, 
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers, 
                            configs.dropout, configs.mlp_activation)
            self.decoder = MLP(self.hidden_dim_of_mlp, self.dec_dim,
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                            configs.dropout, configs.mlp_activation) 
        self.mlp_layer = MLP(self.hidden_dim_of_mlp, self.hidden_dim_of_mlp, 
                             self.hidden_dim_of_mlp, self.hidden_layers_of_mlp)
        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        
        means = x_enc.mean(1, keepdim=True).detach()    
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        bs, _, n_vars = x_enc.shape
        # x_enc: [bs x nvars x seq_len] 
        x_enc = x_enc.permute(0, 2, 1)
        # x_enc: [bs * nvars x seq_len]
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)

        # fold_out: [bs * n_vars x token_num x token_len]
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)

        token_num = fold_out.shape[1]
        # times_embeds: [bs * n_vars x token_num x hidden_dim_of_gpt2]
        times_embeds = self.encoder(fold_out)
        if self.mix:
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            times_embeds_m = times_embeds_m / times_embeds_m.norm(dim=2, keepdim=True)
            times_embeds = times_embeds + self.add_scale * times_embeds_m
        outputs=self.mlp_layer(times_embeds.permute(1, 0, 2)).permute(1, 0, 2)
        # dec_out: [bs * n_vars x token_num x token_len]
        dec_out = self.decoder(outputs)
        state_vars = n_vars * self.ed_ratio
        dec_out = dec_out.reshape(bs, state_vars, -1)
        # dec_out: [bs x token_num * token_len / self.ed_ratio  x state_vars]
        dec_out = dec_out.permute(0, 2, 1)
        
        # dec_out = dec_out * \
        #     (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        # dec_out = dec_out + \
        #     (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        
        return dec_out.to(torch.float32)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)