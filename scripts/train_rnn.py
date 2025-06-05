# scripts/train_rnn.py

import os
import time
import sys
import torch
import torch.nn as nn
import torchinfo # Để đếm tham số

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.tokenizer import CharacterTokenizer
from models.rnn_model import SimpleRNNModel
from prepare_data import QA_PAIRS # Để lấy các cặp Q&A gốc

# Cấu hình đường dẫn
DATA_PATH = './data'
MAX_SEQ_LEN_FILE = os.path.join(DATA_PATH, 'max_seq_len.txt')
TRAINED_RNN_MODEL_PATH = 'result_rnn' # Thư mục riêng cho RNN
MODEL_STATE_DICT_PATH = os.path.join(TRAINED_RNN_MODEL_PATH, 'rnn_model_state_dict.pth')

def load_training_data_for_rnn(data_path, max_seq_len_file, tokenizer):
    """
    Encode từng cặp Q&A, pad chúng về cùng max_seq_len.
    """
    if not os.path.exists(max_seq_len_file):
        raise FileNotFoundError(f" Không tìm thấy file max_seq_len.txt tại '{max_seq_len_file}'")
    with open(max_seq_len_file, 'r') as f:
        max_seq_len = int(f.read().strip())
    print(f" Đã đọc max_seq_len: {max_seq_len} (dùng để padding).")

    encoded_qa_pairs = []
    for qa_pair in QA_PAIRS:
        token_id_list = tokenizer.encode(qa_pair)
        if len(token_id_list) > max_seq_len: # Cắt bớt nếu dài hơn max_seq_len (không nên xảy ra nếu prepare_data đúng)
            token_id_list = token_id_list[:max_seq_len]
        # Pad đến max_seq_len
        padded_token_ids = token_id_list + [tokenizer.pad_token_id] * (max_seq_len - len(token_id_list))
        encoded_qa_pairs.append(torch.tensor([padded_token_ids], dtype=torch.long))

    if not encoded_qa_pairs:
        raise ValueError(" Không có dữ liệu huấn luyện nào sau khi encode. Vui lòng kiểm tra QA_PAIRS và tokenizer.")

    print(f" Đã encode và pad {len(encoded_qa_pairs)} cặp Q&A.")
    return encoded_qa_pairs, max_seq_len


if __name__ == '__main__':
    # --- Cấu hình siêu tham số huấn luyện ---
    num_stages = 50 # Số stages để lưu checkpoint
    epochs_per_stage = 200 # Số epoch cho mỗi stage
    learning_rate = 3e-4

    # --- Cấu hình mô hình RNN ---
    # Điều chỉnh các giá trị này để tìm số lượng tham số tối thiểu
    EMBEDDING_DIM = 4 # Kích thước embedding
    HIDDEN_SIZE = 16  # Kích thước hidden state của LSTM/GRU
    NUM_LAYERS = 1   # Số lượng lớp LSTM/GRU

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Training RNN sẽ chạy trên device: {device}")

    # --- 1. Load tokenizer và chuẩn bị dữ liệu ---
    try:
        tokenizer = CharacterTokenizer.from_pretrained(DATA_PATH)
        print(f" Loaded tokenizer from '{DATA_PATH}' (vocab_size = {tokenizer.vocab_size}).")
    except Exception as e:
        raise RuntimeError(f" Không tìm thấy hoặc không load được tokenizer từ '{DATA_PATH}': {e}")
    
    encoded_qa_pairs, max_seq_len_for_rnn = load_training_data_for_rnn(DATA_PATH, MAX_SEQ_LEN_FILE, tokenizer)
    
    # --- 2. Training loop (staged) ---
    for stage in range(num_stages):
        print(f"\n=== Stage {stage + 1}/{num_stages} (RNN Model) ===")

        # Khởi tạo model và optimizer
        rnn_model = SimpleRNNModel(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            pad_token_id=tokenizer.pad_token_id
        )
        optimizer = torch.optim.AdamW(rnn_model.parameters(), lr=learning_rate)

        # Nếu có checkpoint cũ, load lại
        if os.path.exists(MODEL_STATE_DICT_PATH):
            try:
                ckpt = torch.load(MODEL_STATE_DICT_PATH, map_location=device)
                rnn_model.load_state_dict(ckpt)
                print(f" Loaded checkpoint từ '{MODEL_STATE_DICT_PATH}'.")
            except Exception as e:
                print(f" Lỗi khi load checkpoint cho stage {stage + 1}: {e}")
                print("  Tiếp tục train từ state mới.")

        # In thông tin model và số tham số (chỉ lần đầu)
        if stage == 0:
            print(f"\nCấu hình mô hình RNN: Embedding={EMBEDDING_DIM}, Hidden={HIDDEN_SIZE}, Layers={NUM_LAYERS}")
            print("\nSố lượng tham số mô hình RNN:")
            # Sử dụng một input_size hợp lý để torchinfo tính toán đúng
            torchinfo.summary(rnn_model, input_size=(1, max_seq_len_for_rnn), dtypes=[torch.long])


        # Đưa model lên device và chuyển sang train mode
        rnn_model.to(device)
        rnn_model.train()

        print(f" Bắt đầu train {epochs_per_stage} epochs trong Stage {stage + 1}…")

        # Vòng lặp epoch
        for epoch in range(epochs_per_stage):
            total_loss = 0
            start_time = time.time()

            for i, input_ids_batch in enumerate(encoded_qa_pairs):
                input_ids_batch = input_ids_batch.to(device)

                # RNN forward pass
                # Ở đây labels = input_ids_batch để học language modeling (dự đoán token tiếp theo)
                outputs = rnn_model(input_ids=input_ids_batch, labels=input_ids_batch)
                loss = outputs[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            avg_loss = total_loss / len(encoded_qa_pairs)
            end_time = time.time()
            duration = end_time - start_time

            if (epoch + 1) % 25 == 0 or (epoch + 1) == epochs_per_stage:
                print(f"   Epoch {epoch+1}/{epochs_per_stage} — Avg Loss: {avg_loss:.4f} — Thời gian: {duration:.3f}s")

        print(f" Stage {stage + 1} kết thúc.")

        # Save checkpoint mỗi stage
        try:
            if not os.path.exists(TRAINED_RNN_MODEL_PATH):
                os.makedirs(TRAINED_RNN_MODEL_PATH)
            torch.save(rnn_model.state_dict(), MODEL_STATE_DICT_PATH)
            # Lưu tokenizer cùng với model (không cần thiết mỗi stage nhưng đảm bảo có backup)
            tokenizer.save_pretrained(TRAINED_RNN_MODEL_PATH) 
            print(f" Đã save model state và tokenizer vào '{TRAINED_RNN_MODEL_PATH}'.")
        except Exception as e:
            print(f" Lỗi khi lưu checkpoint Stage {stage + 1}: {e}")

        # Dọn dẹp bộ nhớ
        del rnn_model
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n Hoàn tất tất cả các stage training cho RNN Model!")