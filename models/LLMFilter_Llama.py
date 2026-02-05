import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaModel
from layers.mlp import MLP

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.token_len
        self.enc_dim = configs.token_len
        self.ed_ratio = int(configs.state_dim / configs.obs_dim)
        self.dec_dim = self.ed_ratio * configs.token_len
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)
        
        self.llama = LlamaModel.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float32,
        )
        if configs.llm_ckp_dir == 'model/llama-7b':
            self.hidden_dim_of_llama = 4096
        elif configs.llm_ckp_dir == 'model/llama-13b':
            self.hidden_dim_of_llama = 5120
        else:
            raise ValueError(f"Unknown model directory: {configs.llm_ckp_dir}")
        self.mix = configs.mix_embeds
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))
        
        for name, param in self.llama.named_parameters():
            param.requires_grad = False

        if configs.mlp_hidden_layers == 0:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.enc_dim, self.hidden_dim_of_llama)
            self.decoder = nn.Linear(self.hidden_dim_of_llama, self.dec_dim)
        else:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(self.enc_dim, self.hidden_dim_of_llama, 
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers, 
                            configs.dropout, configs.mlp_activation)
            self.decoder = MLP(self.hidden_dim_of_llama, self.dec_dim,
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                            configs.dropout, configs.mlp_activation) 
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()    
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        bs, _, n_vars = x_enc.shape
        # x_enc: [bs x nvars x seq_len]
        x_enc = x_enc.reshape(bs, -1)
        # x_enc: [bs x nvars * seq_len]
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        # fold_out: [bs * n_vars x token_num x token_len]
        token_num = fold_out.shape[1]
        # times_embeds: [bs * n_vars x token_num x hidden_dim_of_llama]
        times_embeds = self.encoder(fold_out)
        if self.mix:
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
            times_embeds = times_embeds + self.add_scale * x_mark_enc
        # outputs: [bs * n_vars x token_num x hidden_dim_of_llama]
        outputs = self.llama(
            inputs_embeds=times_embeds)[0]
        # dec_out: [bs * n_vars x token_num x token_len]
        dec_out = self.decoder(outputs)
        state_vars = n_vars * self.ed_ratio
        dec_out = dec_out.reshape(bs, state_vars, -1)
        # dec_out: [bs x token_num * token_len / self.ed_ratio  x state_vars]
        dec_out = dec_out.permute(0, 2, 1)
        
        # dec_out = dec_out * \
        #     (stdev[:, 0, :].unsqueeze(1).repeat(1, int(token_num * self.token_len / self.ed_ratio), 1))
        # dec_out = dec_out + \
        #     (means[:, 0, :].unsqueeze(1).repeat(1, int(token_num * self.token_len / self.ed_ratio), 1))

        return dec_out.to(torch.float32)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)