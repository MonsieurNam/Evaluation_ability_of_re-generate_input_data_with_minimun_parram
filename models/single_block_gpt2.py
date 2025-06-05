# File: models/single_block_gpt2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPT2Config:
    """
    Configuration class for GPT-2 models, manually defining standard configurations.
    """
    def __init__(self, vocab_size=None, n_positions=None): # Thêm n_positions

        # Sử dụng n_positions được cung cấp hoặc giá trị mặc định rất lớn nếu không có
        # Trong trường hợp của chúng ta, nó sẽ được cung cấp từ data preparation.
        self.n_positions = n_positions if n_positions is not None else 1024
        self.n_embd = 4 # Có thể điều chỉnh để tìm param tối thiểu
        self.n_layer = 1
        self.n_head = 1 # n_head phải là ước của n_embd, 1 là an toàn
        self.embd_pdrop = 0.1
        self.attn_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.layer_norm_epsilon = 1e-5
        self.initializer_range = 0.02
        self.vocab_size = vocab_size if vocab_size is not None else 50257

        self.scale_attn_weights = True
        self.use_cache = False

    def __repr__(self):
        return f"GPT2Config(vocab_size={self.vocab_size}, n_positions={self.n_positions}, n_embd={self.n_embd}, n_layer={self.n_layer}, n_head={self.n_head})"

def NewGELUActivation(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.split_size = self.n_embd
        # Sử dụng config.n_positions cho bias
        self.register_buffer("bias", torch.tril(torch.ones((config.n_positions, config.n_positions), dtype=torch.uint8)).view(1, 1, config.n_positions, config.n_positions))

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, q, k, v, attention_mask=None):
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        attn_weights = attn_weights / (float(v.size(-1)) ** 0.5)

        query_length, key_length = q.size(-2), k.size(-2)
        # Sử dụng kích thước động cho causal_mask
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
        attn_weights = torch.where(causal_mask, attn_weights, torch.tensor(-1e4, dtype=attn_weights.dtype, device=attn_weights.device))

        if attention_mask is not None:
             attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        return attn_output, attn_weights

    def forward(self, hidden_states, attention_mask=None):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = query.view(query.size(0), query.size(1), self.n_head, self.split_size // self.n_head).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), self.n_head, self.split_size // self.n_head).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.n_head, self.split_size // self.n_head).transpose(1, 2)

        attn_output, attn_weights = self._attn(query, key, value, attention_mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(attn_output.size(0), -1, self.n_embd)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.n_embd
        inner_dim = embed_dim * 4

        self.c_fc = nn.Linear(embed_dim, inner_dim)
        self.c_proj = nn.Linear(inner_dim, embed_dim)
        self.act = NewGELUActivation
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = hidden_size * 4

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs, attn_weights = self.attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + attn_outputs

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states, attn_weights

class SingleBlockGPT2ModelNoDepend(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd) # n_positions từ config
        self.drop = nn.Dropout(config.embd_pdrop)

        self.h = TransformerBlock(config)

        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids, position_ids=None, attention_mask=None, labels=None):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Đảm bảo position_ids không vượt quá n_positions
        if position_ids.max() >= self.config.n_positions:
            raise ValueError(
                f"Position IDs (max: {position_ids.max().item()}) vượt quá "
                f"n_positions được cấu hình ({self.config.n_positions}). "
                "Vui lòng kiểm tra lại độ dài chuỗi đầu vào hoặc cấu hình mô hình."
            )

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        hidden_states, attn_weights = self.h(
            hidden_states,
            attention_mask=attention_mask,
        )

        hidden_states = self.ln_f(hidden_states)
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        output = (lm_logits,)
        return ((loss,) + output) if loss is not None else output

    def generate(self, input_ids, max_length, pad_token_id, eos_token_id=None, temperature=1.0, top_k=None, top_p=None, device='cpu'):
        self.eval()
        input_ids = input_ids.to(device)
        generated_ids = input_ids.clone()

        for _ in range(max_length - input_ids.shape[-1]):
            with torch.no_grad():
                # Lấy output của token cuối cùng từ sequence hiện tại
                outputs = self(generated_ids)
                logits = outputs[0][:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('inf')

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('inf')

                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).squeeze(1)

            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(-1)], dim=-1)

            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            
            # Kiểm tra nếu độ dài sinh ra đã đạt n_positions của mô hình
            if generated_ids.shape[1] >= self.config.n_positions:
                break # Dừng sinh nếu đạt đến độ dài tối đa mà mô hình có thể xử lý (n_positions)

        return generated_ids