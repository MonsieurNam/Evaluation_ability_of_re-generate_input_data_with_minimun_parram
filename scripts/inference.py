# File: inference.py

import os
import torch
import sys

# Thêm project root vào sys.path
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.tokenizer import CharacterTokenizer
from models.single_block_gpt2 import SingleBlockGPT2ModelNoDepend, GPT2Config
from prepare_data import QA_PAIRS  # Import QA_PAIRS từ prepare_data.py

# Thư mục chứa model và tokenizer đã train xong
TRAINED_SINGLE_BLOCK_MODEL_PATH = 'results'
MODEL_STATE_DICT_PATH = os.path.join(TRAINED_SINGLE_BLOCK_MODEL_PATH, 'single_block_model_state_dict.pth')
DATA_PATH = './data'  # Thư mục chứa data.txt, vocab.json, max_seq_len.txt
MAX_SEQ_LEN_FILE = os.path.join(DATA_PATH, 'max_seq_len.txt')


def load_trained_single_block_model(model_dir: str, state_dict_path: str, max_seq_len_file: str, device: torch.device):
    """
    - Load tokenizer từ model_dir.
    - Đọc max_seq_len từ file.
    - Tạo config dựa trên vocab_size của tokenizer và max_seq_len.
    - Khởi tạo SingleBlockGPT2ModelNoDepend với config đó.
    - Load state_dict từ state_dict_path.
    - Trả về (model, tokenizer, max_seq_len).
    """
    # 1. Load tokenizer
    try:
        tokenizer = CharacterTokenizer.from_pretrained(model_dir)
        print(f"Loaded tokenizer from '{model_dir}' (vocab_size = {tokenizer.vocab_size}).")
    except Exception as e:
        print(f"Error loading tokenizer from '{model_dir}': {e}")
        return None, None, None

    # 2. Đọc max_seq_len từ file
    if not os.path.exists(max_seq_len_file):
        print(f"Không tìm thấy file max_seq_len.txt tại '{max_seq_len_file}'.")
        return None, None, None
    with open(max_seq_len_file, 'r') as f:
        max_seq_len = int(f.read().strip())
    print(f"🔢 Đã đọc max_seq_len: {max_seq_len} (sẽ dùng làm n_positions cho model).")

    # 3. Tạo config dùng vocab_size và n_positions vừa load
    config = GPT2Config(vocab_size=tokenizer.vocab_size, n_positions=max_seq_len)

    # 4. Khởi tạo model
    model = SingleBlockGPT2ModelNoDepend(config)

    # 5. Load state_dict (checkpoint đã train)
    if not os.path.exists(state_dict_path):
        print(f"Không tìm thấy file state_dict tại '{state_dict_path}'.")
        return None, None, None

    try:
        ckpt = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(ckpt)
        print(f"Loaded model state_dict from '{state_dict_path}'.")
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        return None, None, None

    # 6. Đưa model về đúng device và set eval mode
    model.to(device)
    model.eval()

    return model, tokenizer, max_seq_len


def count_model_params(model: torch.nn.Module):
    """
    Đếm tổng số tham số (parameters) và số tham số trainable (requires_grad=True).
    Trả về (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def generate_text(
    model: SingleBlockGPT2ModelNoDepend,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_length: int,  # Tổng số token đầu ra (bao gồm prompt)
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device
) -> str:
    """
    - Mô tả: sinh văn bản dựa trên model đã train và tokenizer.
    - prompt: chuỗi đầu vào (ví dụ: "Question: Xin chào").
    """
    token_id_list = tokenizer.encode(prompt)
    input_ids = torch.tensor([token_id_list], dtype=torch.long, device=device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device
        )

    generated_ids_list = generated_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids_list)

    return generated_text


if __name__ == '__main__':
    # 1. Thiết lập device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶️ Inference sẽ chạy trên device: {device}")

    # 2. Load model + tokenizer + max_seq_len đã train
    model, tokenizer, max_seq_len_from_training = load_trained_single_block_model(
        TRAINED_SINGLE_BLOCK_MODEL_PATH,
        MODEL_STATE_DICT_PATH,
        MAX_SEQ_LEN_FILE,
        device
    )

    if model is None or tokenizer is None:
        print("Failed to load model/tokenizer. Kết thúc inference.")
        exit(1)

    # 2.1. In số lượng tham số của model
    total_params, trainable_params = count_model_params(model)
    print(f"Model có tổng cộng {total_params:,} tham số, trong đó {trainable_params:,} tham số trainable.\n")

    # 3. Định nghĩa các cặp prompt và full_target tương ứng
    #    Để tái tạo input chính xác, chúng ta cần biết full_target của từng prompt.
    test_cases = []
    for qa_pair in QA_PAIRS:
        parts = qa_pair.split('\nAnswer:', 1)
        if len(parts) == 2:
            prompt_q = parts[0].strip()  # "Question: Xin chào"
            full_target_qa = qa_pair.strip()  # "Question: Xin chào \nAnswer: FPT University xin chào bạn!."
            test_cases.append({
                'prompt': prompt_q,
                'full_target': full_target_qa,
                'target_length': len(tokenizer.encode(full_target_qa))
            })
        else:
            # Xử lý các trường hợp không phải Q&A nếu có (ví dụ: "Python is a programming language.")
            # Coi cả câu là prompt và target nếu không có \nAnswer:
            test_cases.append({
                'prompt': qa_pair.strip(),
                'full_target': qa_pair.strip(),
                'target_length': len(tokenizer.encode(qa_pair.strip()))
            })

    # 4. Các tham số sinh văn bản được điều chỉnh để ưu tiên tái tạo chính xác
    temperature = 0.01  # Rất thấp, gần greedy decoding
    top_k = 1  # Luôn chọn token có xác suất cao nhất
    top_p = 0.95  # Sẽ không ảnh hưởng nhiều khi top_k=1 và temperature thấp

    print(f"\n--- Bắt đầu Kiểm tra Khả năng Tái tạo (Reproduction) ---")
    print(f"Tham số sinh văn bản: temperature={temperature}, top_k={top_k}, top_p={top_p}\n")

    for i, test_case in enumerate(test_cases):
        input_prompt = test_case['prompt']
        full_target_text = test_case['full_target']
        max_gen_length = test_case['target_length']  # Đảm bảo sinh đủ độ dài của cặp QA đã học

        print(f"\n--- Test Case {i+1}/{len(test_cases)} ---")
        print(f"  Prompt: \"{input_prompt}\"")
        print(f"  Expected full target length: {max_gen_length} tokens "
              f"(from '{full_target_text}')")
        print(f"  Generating up to {max_gen_length} tokens...\n")

        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=input_prompt,
            max_length=max_gen_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device
        )

        print("  🔹 Generated Text:")
        print("  -----------------------------------------------------------")
        print(generated)
        print("  -----------------------------------------------------------")

        # Đánh giá độ chính xác tái tạo
        if generated == full_target_text:
            print("Tái tạo HOÀN TOÀN chính xác!")
        else:
            print("Tái tạo KHÔNG chính xác.")
            print(f"  Target:   '{full_target_text}'")
            print(f"  Generated:'{generated}'")
            # Bạn có thể thêm logic tính token accuracy ở đây nếu muốn chi tiết hơn

    print("\n--- Kiểm tra Khả năng Tái tạo Hoàn tất ---")

    # 5. Dọn dẹp
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
