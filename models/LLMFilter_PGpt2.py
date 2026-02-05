import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer
from layers.mlp import MLP
from utils.tools import load_content
import numpy as np
import transformers
transformers.logging.set_verbosity_error()

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = int(configs.window_length / 2)
        self.enc_dim = self.token_len
        self.ed_ratio = int(configs.state_dim / configs.obs_dim)
        self.dec_dim = self.ed_ratio * self.token_len
        self.prompt_domain = configs.prompt_domain
        
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)
        self.gpt = GPT2Model.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float32,
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float32,
        )
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
            
        self.hidden_dim_of_gpt2 = 768
        
        # Optionally enable embedding mixing with a scaling parameter
        self.mix = configs.mix_embeds
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))
        
        # Set padding token for the tokenizer
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # Freeze LLaMA model parameters to avoid updating during training
        for name, param in self.gpt.named_parameters():
            param.requires_grad = False

        # Load domain-specific description or use a default one
        if self.prompt_domain:
            self.description = load_content(configs)  # Load custom content based on configurations
        else:
            self.description = 'This two-dimensional velocity system is a fundamental kinematic model.'

        # Determine whether to use linear layers or MLPs for encoder and decoder
        if configs.mlp_hidden_layers == 0:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("Using linear layers as tokenizer and detokenizer")
            # Linear layers for tokenization and de-tokenization
            self.encoder = nn.Linear(self.enc_dim, self.hidden_dim_of_gpt2)
            self.decoder = nn.Linear(self.hidden_dim_of_gpt2, self.dec_dim)
        else:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("Using MLPs as tokenizer and detokenizer")
            # Multi-Layer Perceptrons (MLPs) for tokenization and de-tokenization
            self.encoder = MLP(
                self.enc_dim, self.hidden_dim_of_gpt2, 
                configs.mlp_hidden_dim, configs.mlp_hidden_layers, 
                configs.dropout, configs.mlp_activation
            )
            self.decoder = MLP(
                self.hidden_dim_of_gpt2, self.dec_dim,
                configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                configs.dropout, configs.mlp_activation
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Perform forecasting using the encoder and LLaMA model with optional prompt-based adaptation.

        Args:
            x_enc (Tensor): Input features for the encoder. Shape: [batch_size, seq_len, n_vars].
            x_mark_enc (Tensor): Temporal or auxiliary information for the encoder. Shape: [batch_size, seq_len, n_vars].
            x_dec (Tensor): Decoder input features (not used in this method).
            x_mark_dec (Tensor): Temporal or auxiliary information for the decoder (not used).

        Returns:
            dec_out (Tensor): Forecasted outputs in the desired format. Shape: [batch_size, seq_len/state_vars, state_vars].
        """
        # Normalize encoder inputs
        means = x_enc.mean(1, keepdim=True).detach()  # Shape: [batch_size, 1, n_vars]
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # Shape: [batch_size, 1, n_vars]
        x_enc /= stdev

        # Prepare batch and sequence dimensions
        bs, seq_len, n_vars = x_enc.shape
        
        # Reshape and fold encoder inputs into tokens
        x_enc = x_enc.reshape(bs, -1)  # Shape: [batch_size, seq_len * n_vars]
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)  # Shape: [batch_size, token_num, token_len]
        token_num = fold_out.shape[1]

        # Encode tokens
        input_embeds = self.encoder(fold_out)  # Shape: [batch_size, token_num, hidden_dim_of_gpt2]

        # Prompt-based adaptation during inference
        input_embeds = self.apply_prompt(input_embeds, bs)

        # Pass through the LLaMA model
        outputs = self.gpt(
            inputs_embeds=input_embeds)[0]  # Shape: [batch_size, prompt_token_len + token_num, hidden_dim_of_gpt2]

        # Decode outputs
        dec_out = self.decoder(outputs[:, -token_num:, :])  # Shape: [batch_size, token_num, token_len]
        state_vars = n_vars * self.ed_ratio
        dec_out = dec_out.reshape(bs, state_vars, -1)  # Shape: [batch_size, state_vars, token_len / ed_ratio]
        dec_out = dec_out.permute(0, 2, 1)  # Shape: [batch_size, token_len / ed_ratio, state_vars]

        # Return forecasted outputs
        return dec_out.to(torch.float32)

    def generate_prompt_embeddings(self, bs):
        """
        Generate prompt embeddings for the given batch size.

        Args:
            bs (int): Batch size.

        Returns:
            prompt_embeddings (Tensor): Generated prompt embeddings. Shape: [batch_size, prompt_token_len, hidden_dim_of_gpt2].
        """
        # Generate prompt tokens
        prompt = [
            f"<|start_prompt|>{self.description}<|<end_prompt>|>"
            for _ in range(bs)
        ]
        prompt = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).input_ids
        prompt = prompt.reshape(bs, -1)  # Shape: [batch_size, prompt_token_len]
        prompt_embeddings = self.gpt.get_input_embeddings()(prompt.to(self.device))  # Shape: [batch_size, prompt_token_len, hidden_dim_of_gpt2]
        self.prompt_domain = False 

        return prompt_embeddings

    def apply_prompt(self, input_embeds, batch_size):
        """
        Applies prompt embeddings to input embeddings during inference if prompt-based adaptation is enabled.

        Args:
            input_embeds (Tensor): The input embeddings. Shape: [batch_size, token_num, hidden_dim_of_gpt2].
            batch_size (int): The batch size.

        Returns:
            Tensor: Input embeddings with prompt embeddings prepended if applicable.
        """
        if self.prompt_domain and not self.training:
            prompt_embeddings = self.generate_prompt_embeddings(batch_size)
            input_embeds = torch.cat([prompt_embeddings, input_embeds], dim=1)  # Combine prompt and input embeddings
            # print("Prompt applied during inference.")
        return input_embeds
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forward method to interface with the forecasting function.
        
        Args:
            x_enc (Tensor): Input features for the encoder.
            x_mark_enc (Tensor): Temporal or auxiliary information for the encoder.
            x_dec (Tensor): Decoder input features (not used).
            x_mark_dec (Tensor): Temporal or auxiliary information for the decoder (not used).

        Returns:
            Output from the forecasting method.
        """
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
