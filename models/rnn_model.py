# models/rnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys

# Thêm project root vào sys.path để import tokenizer
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.tokenizer import CharacterTokenizer

class SimpleRNNModel(nn.Module):
    """
    Một mô hình RNN đơn giản sử dụng LSTM để dự đoán token tiếp theo.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int, pad_token_id: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        # Sử dụng LSTM (bạn có thể thay đổi thành nn.GRU nếu muốn)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

        # Khởi tạo trọng số (tùy chọn, có thể giúp hội tụ nhanh hơn)
        self.init_weights()

    def init_weights(self):
        # Khởi tạo trọng số cho embedding layer
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        # Khởi tạo trọng số cho lớp linear cuối
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, input_ids, hidden=None, labels=None):
        # input_ids: (batch_size, seq_len)
        embedded = self.embedding(input_ids) # (batch_size, seq_len, embedding_dim)

        # hidden là tuple (h_n, c_n) cho LSTM
        output, hidden = self.rnn(embedded, hidden) # output: (batch_size, seq_len, hidden_size)

        # Ánh xạ output của RNN về kích thước từ vựng
        logits = self.fc(output) # (batch_size, seq_len, vocab_size)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # Loại bỏ token cuối cùng từ logits và token đầu tiên từ labels
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id) # Bỏ qua pad_token_id khi tính loss
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return (loss, logits) if loss is not None else (logits,)

    def generate(self, input_ids, max_length, pad_token_id, temperature=1.0, top_k=None, top_p=None, device='cpu'):
        """
        Sinh văn bản token by token.
        input_ids: tensor (1, current_seq_len)
        max_length: tổng số token tối đa cần sinh (bao gồm input_ids ban đầu)
        """
        self.eval() # Chuyển sang chế độ đánh giá
        generated_ids = input_ids.clone().to(device)
        current_hidden = None # Trạng thái ẩn ban đầu

        with torch.no_grad():
            for _ in range(max_length - generated_ids.shape[-1]):
                # Chỉ truyền token cuối cùng vào mô hình để dự đoán token tiếp theo
                # Điều này mô phỏng quá trình sinh văn bản autoregressive
                current_input_token = generated_ids[:, -1].unsqueeze(0) # (1, 1)
                
                # Forward pass với trạng thái ẩn hiện tại
                output, current_hidden = self.rnn(self.embedding(current_input_token), current_hidden)
                
                # Lấy logits từ output của token hiện tại
                logits = self.fc(output.squeeze(1)) / temperature # (1, vocab_size)

                # Áp dụng top-k, top-p sampling (simplified)
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('inf')

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('inf')

                probs = F.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).squeeze(1)

                generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(-1)], dim=-1)
                
                # Dừng nếu độ dài đã đạt max_length
                if generated_ids.shape[1] >= max_length:
                    break

        return generated_ids

# # Ví dụ sử dụng
# if __name__ == '__main__':
#     DATA_PATH = '../data'
#     try:
#         tokenizer = CharacterTokenizer.from_pretrained(DATA_PATH)
#     except Exception as e:
#         print(f"Lỗi khi load tokenizer: {e}. Vui lòng chạy prepare_data.py trước.")
#         sys.exit(1)
    
#     vocab_size = tokenizer.vocab_size
#     pad_token_id = tokenizer.pad_token_id
    
#     # Cấu hình nhỏ để test
#     embedding_dim = 8
#     hidden_size = 16
#     num_layers = 1
    
#     model = SimpleRNNModel(vocab_size, embedding_dim, hidden_size, num_layers, pad_token_id)
#     print(model)

#     # Input giả định
#     dummy_input = torch.randint(0, vocab_size, (1, 10)) # Batch size 1, seq len 10
    
#     # Test forward pass
#     output_logits = model(dummy_input)[0]
#     print(f"Output logits shape: {output_logits.shape}") # Should be (1, 10, vocab_size)

#     # Test generation
#     prompt_ids = torch.tensor([[tokenizer.encode("Hello")[0]]], dtype=torch.long) # Bắt đầu với 1 token
#     generated_seq = model.generate(prompt_ids, max_length=20, pad_token_id=pad_token_id)
#     print(f"Generated sequence (ids): {generated_seq}")
#     print(f"Generated text: '{tokenizer.decode(generated_seq[0].tolist())}'")