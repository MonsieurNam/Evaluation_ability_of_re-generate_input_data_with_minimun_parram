# scripts/inference_rnn.py

import os
import torch
import sys
import time

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
TRAINED_RNN_MODEL_PATH = 'result_rnn' # Thư mục chứa model và tokenizer đã train xong
MODEL_STATE_DICT_PATH = os.path.join(TRAINED_RNN_MODEL_PATH, 'rnn_model_state_dict.pth')


def load_trained_rnn_model(model_dir: str, state_dict_path: str, max_seq_len_file: str, device: torch.device):
    """
    - Load tokenizer từ model_dir.
    - Đọc max_seq_len từ file.
    - Khởi tạo SimpleRNNModel với cấu hình đã dùng khi train.
    - Load state_dict từ state_dict_path.
    - Trả về (model, tokenizer, max_seq_len).
    """
    # 1. Load tokenizer
    try:
        tokenizer = CharacterTokenizer.from_pretrained(model_dir)
        print(f" Loaded tokenizer from '{model_dir}' (vocab_size = {tokenizer.vocab_size}).")
    except Exception as e:
        print(f" Error loading tokenizer from '{model_dir}': {e}")
        return None, None, None

    # 2. Đọc max_seq_len
    if not os.path.exists(max_seq_len_file):
        print(f" Không tìm thấy file max_seq_len.txt tại '{max_seq_len_file}'.")
        return None, None, None
    with open(max_seq_len_file, 'r') as f:
        max_seq_len = int(f.read().strip())
    print(f" Đã đọc max_seq_len: {max_seq_len} (dùng để padding và max_length khi sinh).")

    # 3. Khởi tạo model với cấu hình phù hợp (PHẢI TRÙNG VỚI CẤU HÌNH KHI TRAIN)
    #    Để thực tế, bạn có thể lưu cấu hình vào một file JSON khi train,
    #    sau đó đọc lại ở đây. Hiện tại, ta sẽ giả định cấu hình này
    #    trùng với cấu hình trong train_rnn.py.
    #    Nếu muốn thay đổi, hãy thay đổi trong cả 2 file.
    EMBEDDING_DIM = 4
    HIDDEN_SIZE = 16
    NUM_LAYERS = 1

    model = SimpleRNNModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        pad_token_id=tokenizer.pad_token_id
    )

    # 4. Load state_dict
    if not os.path.exists(state_dict_path):
        print(f" Không tìm thấy file state_dict tại '{state_dict_path}'.")
        return None, None, None

    try:
        ckpt = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(ckpt)
        print(f" Loaded model state_dict from '{state_dict_path}'.")
    except Exception as e:
        print(f" Error loading state_dict: {e}")
        return None, None, None

    # 5. Đưa model về đúng device và set eval mode
    model.to(device)
    model.eval()

    return model, tokenizer, max_seq_len


def calculate_accuracy(generated_text: str, target_text: str, tokenizer: CharacterTokenizer):
    """Tính token accuracy và exact match."""
    generated_ids = tokenizer.encode(generated_text)
    target_ids = tokenizer.encode(target_text)

    # Đảm bảo độ dài bằng nhau để so sánh
    min_len = min(len(generated_ids), len(target_ids))
    correct_tokens = 0
    for i in range(min_len):
        if generated_ids[i] == target_ids[i]:
            correct_tokens += 1
    
    token_accuracy = (correct_tokens / len(target_ids)) * 100 if len(target_ids) > 0 else 0
    exact_match = (generated_text == target_text)

    return token_accuracy, exact_match


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference RNN sẽ chạy trên device: {device}")

    # Load model + tokenizer + max_seq_len đã train
    model, tokenizer, max_seq_len_from_training = load_trained_rnn_model(
        TRAINED_RNN_MODEL_PATH,
        MODEL_STATE_DICT_PATH,
        MAX_SEQ_LEN_FILE,
        device
    )

    if model is None or tokenizer is None:
        print("Failed to load model/tokenizer. Kết thúc inference.")
        exit(1)

    # Chuẩn bị test cases từ QA_PAIRS (tương tự prepare_data và evaluate_baselines)
    test_cases = []
    for qa_pair in QA_PAIRS:
        parts = qa_pair.split('\nAnswer:', 1)
        if len(parts) == 2:
            prompt_q = parts[0].strip()
            full_target_qa = qa_pair.strip()
            test_cases.append({
                'prompt': prompt_q,
                'full_target': full_target_qa,
                'target_length': len(tokenizer.encode(full_target_qa))
            })
        else:
            test_cases.append({
                'prompt': qa_pair.strip(),
                'full_target': qa_pair.strip(),
                'target_length': len(tokenizer.encode(qa_pair.strip()))
            })

    # Các tham số sinh văn bản được điều chỉnh để ưu tiên tái tạo chính xác
    temperature = 0.01
    top_k = 1
    top_p = 0.95

    print(f"\n--- Bắt đầu Kiểm tra Khả năng Tái tạo (RNN Model) ---")
    print(f"Tham số sinh văn bản: temperature={temperature}, top_k={top_k}, top_p={top_p}\n")
    
    total_exact_matches = 0
    total_inference_time_ms = 0

    for i, test_case in enumerate(test_cases):
        input_prompt = test_case['prompt']
        full_target_text = test_case['full_target']
        # Đảm bảo max_gen_length không vượt quá max_seq_len mà model được train
        # Tuy nhiên, nếu model được train trên từng cặp QA, thì max_gen_length chính là target_length
        # của cặp QA đó.
        max_gen_length = test_case['target_length'] 

        print(f"\n--- Test Case {i+1}/{len(test_cases)} ---")
        print(f"  Prompt: \"{input_prompt}\"")
        print(f"  Expected full target length: {max_gen_length} tokens (from '{full_target_text}')")
        print(f"  Generating up to {max_gen_length} tokens...\n")

        start_time = time.time()
        generated = model.generate(
            input_ids=torch.tensor([tokenizer.encode(input_prompt)], dtype=torch.long),
            max_length=max_gen_length,
            pad_token_id=tokenizer.pad_token_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device
        )
        end_time = time.time()
        total_inference_time_ms += (end_time - start_time) * 1000

        print("  🔹 Generated Text:")
        print("  -----------------------------------------------------------")
        print(generated)
        print("  -----------------------------------------------------------")

        # Đánh giá độ chính xác tái tạo
        token_acc, exact_match = calculate_accuracy(generated, full_target_text, tokenizer)
        if exact_match:
            total_exact_matches += 1
            print("Tái tạo HOÀN TOÀN chính xác!")
        else:
            print(f"Tái tạo KHÔNG chính xác (Token Accuracy: {token_acc:.2f}%).")
            print(f"Target:   '{full_target_text}'")
            print(f"Generated:'{generated}'")
    
    avg_inference_time_ms = total_inference_time_ms / len(test_cases)
    exact_match_rate = (total_exact_matches / len(test_cases)) * 100

    print(f"\n--- Tóm tắt Đánh giá RNN Model ---")
    print(f"  Tổng số test cases: {len(test_cases)}")
    print(f"  Số lượng tái tạo chính xác hoàn toàn: {total_exact_matches}")
    print(f"  Exact Match Rate: {exact_match_rate:.2f}%")
    print(f"  Thời gian suy luận trung bình: {avg_inference_time_ms:.3f} ms/chuỗi")

    # TODO: Lưu kết quả vào file CSV (khi đã hoàn thành cả Transformer)
    # Để tránh ghi đè, ta sẽ lưu kết quả của RNN và Transformer vào một file chung sau Giai đoạn 4.

    # Dọn dẹp
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()