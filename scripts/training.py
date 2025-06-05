# File: training.py

import os
import time
import sys
import torch
import torch.nn as nn
import torchinfo # Để đếm tham số dễ dàng

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.tokenizer import CharacterTokenizer
from models.single_block_gpt2 import SingleBlockGPT2ModelNoDepend, GPT2Config
from prepare_data import QA_PAIRS # Import QA_PAIRS từ prepare_data để có danh sách các cặp Q&A

# Thư mục chứa vocab.json, data.txt và max_seq_len.txt
DATA_PATH = './data'
MAX_SEQ_LEN_FILE = os.path.join(DATA_PATH, 'max_seq_len.txt')

# Thư mục lưu kết quả training
TRAINED_SINGLE_BLOCK_MODEL_PATH = 'results'
MODEL_STATE_DICT_PATH = os.path.join(TRAINED_SINGLE_BLOCK_MODEL_PATH, 'single_block_model_state_dict.pth')


def load_training_data(data_path, max_seq_len_file):
    """
    1. Load tokenizer từ data_path.
    2. Đọc max_seq_len từ file.
    3. Encode từng cặp Q&A trong QA_PAIRS thành list token IDs.
    4. Trả về tokenizer, list các tensor encoded, và max_seq_len.
    """
    # -- 1. Load tokenizer --
    try:
        tokenizer = CharacterTokenizer.from_pretrained(data_path)
        print(f" Loaded tokenizer from '{data_path}' (vocab_size = {tokenizer.vocab_size}).")
    except Exception as e:
        raise RuntimeError(f" Không tìm thấy hoặc không load được tokenizer từ '{data_path}': {e}")

    # -- 2. Đọc max_seq_len --
    if not os.path.exists(max_seq_len_file):
        raise FileNotFoundError(f" Không tìm thấy file max_seq_len.txt tại '{max_seq_len_file}'")
    with open(max_seq_len_file, 'r') as f:
        max_seq_len = int(f.read().strip())
    print(f" Đã đọc max_seq_len: {max_seq_len}")

    # -- 3. Encode từng cặp Q&A --
    encoded_qa_pairs = []
    for qa_pair in QA_PAIRS: # Sử dụng QA_PAIRS từ prepare_data.py
        token_id_list = tokenizer.encode(qa_pair)
        if len(token_id_list) == 0:
            print(f" Cảnh báo: Một cặp QA trống sau khi encode: '{qa_pair}'")
            continue
        # Chuyển thành tensor shape [1, seq_len]
        encoded_qa_pairs.append(torch.tensor([token_id_list], dtype=torch.long))

    if not encoded_qa_pairs:
        raise ValueError(" Không có dữ liệu huấn luyện nào sau khi encode. Vui lòng kiểm tra QA_PAIRS và tokenizer.")

    print(f" Đã encode {len(encoded_qa_pairs)} cặp Q&A.")

    return tokenizer, encoded_qa_pairs, max_seq_len


if __name__ == '__main__':
    # -- 0. Chuẩn bị: số stage, epoch, device --
    num_stages = 100 # Số stages (lưu checkpoint mỗi stage)
    epochs_per_stage = 200 # Số epoch cho mỗi stage (có thể cần nhiều hơn để học thuộc từng cặp)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶️ Training sẽ chạy trên device: {device}")

    # -- 1. Load tokenizer, encoded data và max_seq_len chỉ một lần duy nhất --
    tokenizer, encoded_qa_pairs, max_seq_len_for_model = load_training_data(DATA_PATH, MAX_SEQ_LEN_FILE)
    print(f" Model sẽ được cấu hình với n_positions = {max_seq_len_for_model}")

    # -- 2. Training loop (staged) --
    for stage in range(num_stages):
        print(f"\n=== Stage {stage + 1}/{num_stages} ===")

        # 2.1 Tạo config mới dựa trên vocab_size và max_seq_len
        # max_seq_len_for_model sẽ là n_positions của mô hình
        config = GPT2Config(vocab_size=tokenizer.vocab_size, n_positions=max_seq_len_for_model)

        # 2.2 Khởi tạo model + optimizer
        single_block_model = SingleBlockGPT2ModelNoDepend(config)
        optimizer = torch.optim.AdamW(single_block_model.parameters(), lr=3e-4)

        # 2.3 Nếu có checkpoint cũ, load lại
        if os.path.exists(MODEL_STATE_DICT_PATH):
            try:
                ckpt = torch.load(MODEL_STATE_DICT_PATH, map_location=device)
                single_block_model.load_state_dict(ckpt)
                print(f" Loaded checkpoint từ '{MODEL_STATE_DICT_PATH}'.")
            except Exception as e:
                print(f" Lỗi khi load checkpoint cho stage {stage + 1}: {e}")
                print("  Tiếp tục train từ state mới.")

        # 2.4 In thông tin model và số tham số (chỉ lần đầu)
        if stage == 0:
            print("\nCấu hình mô hình:")
            print(config)
            print("\nSố lượng tham số mô hình:")
            torchinfo.summary(single_block_model, input_size=(1, config.n_positions), dtypes=[torch.long])


        # 2.5 Đưa model lên device và chuyển sang train mode
        single_block_model.to(device)
        single_block_model.train()

        print(f" Bắt đầu train {epochs_per_stage} epochs trong Stage {stage + 1}…")

        # 2.6 Vòng lặp epoch
        for epoch in range(epochs_per_stage):
            total_loss = 0
            start_time = time.time()

            # Lặp qua từng cặp Q&A trong dataset
            for i, input_ids_batch in enumerate(encoded_qa_pairs):
                # Đưa input_ids_batch lên device
                input_ids_batch = input_ids_batch.to(device)

                # Forward pass: pass input_ids_batch làm cả input và labels
                # model trả về tuple (loss, logits, ...)
                outputs = single_block_model(input_ids=input_ids_batch, labels=input_ids_batch)
                loss = outputs[0]

                # Backward + optimize (có thể tích lũy gradient nếu muốn batch lớn hơn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            avg_loss = total_loss / len(encoded_qa_pairs)
            end_time = time.time()
            duration = end_time - start_time

            # In progress mỗi 25 epoch hoặc cuối epoch
            if (epoch + 1) % 25 == 0 or (epoch + 1) == epochs_per_stage:
                print(f"   Epoch {epoch+1}/{epochs_per_stage} — Avg Loss: {avg_loss:.4f} — Thời gian: {duration:.3f}s")

        print(f" Stage {stage + 1} kết thúc.")

        # -- 2.7 Save checkpoint mỗi stage --
        try:
            if not os.path.exists(TRAINED_SINGLE_BLOCK_MODEL_PATH):
                os.makedirs(TRAINED_SINGLE_BLOCK_MODEL_PATH)
            torch.save(single_block_model.state_dict(), MODEL_STATE_DICT_PATH)
            tokenizer.save_pretrained(TRAINED_SINGLE_BLOCK_MODEL_PATH)
            print(f" Đã save model state và tokenizer vào '{TRAINED_SINGLE_BLOCK_MODEL_PATH}'.")
        except Exception as e:
            print(f" Lỗi khi lưu checkpoint Stage {stage + 1}: {e}")

        # 2.8 Dọn dẹp bộ nhớ
        del single_block_model
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n Hoàn tất tất cả các stage training!")