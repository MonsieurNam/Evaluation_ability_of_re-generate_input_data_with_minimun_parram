# File: prepare_data.py
import os
import sys

# Thêm project root vào sys.path nếu cần (tùy thuộc vào cách bạn chạy script này)
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.tokenizer import CharacterTokenizer

# Định nghĩa các cặp Q&A (dữ liệu bạn đã cung cấp)
QA_PAIRS = [
    "Question: Xin chào \nAnswer: FPT University xin chào bạn!."
]

DATA_DIR = './data'
DATA_FILE = os.path.join(DATA_DIR, 'data.txt')
MAX_SEQ_LEN_FILE = os.path.join(DATA_DIR, 'max_seq_len.txt') # File để lưu độ dài chuỗi lớn nhất


def prepare_qa_data():
    """
    1. Tạo thư mục data nếu chưa có.
    2. Ghi các cặp Q&A vào data.txt, mỗi cặp cách nhau bởi '\n\n'.
    3. Huấn luyện CharacterTokenizer trên toàn bộ văn bản.
    4. Lưu tokenizer.
    5. Tính toán và lưu độ dài chuỗi mã hóa lớn nhất trong QA_PAIRS.
    """
    print("--- Chuẩn bị Dữ liệu Q&A ---")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f" Đã tạo thư mục: {DATA_DIR}")

    # Ghi các cặp Q&A vào data.txt
    text_content = "\n\n".join(QA_PAIRS) # Nối các cặp bằng 2 ký tự xuống dòng
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        f.write(text_content)
    print(f" Đã ghi {len(QA_PAIRS)} cặp Q&A vào '{DATA_FILE}'.")

    # Huấn luyện tokenizer trên toàn bộ văn bản từ QA_PAIRS
    tokenizer = CharacterTokenizer()
    tokenizer.train(text_content)
    tokenizer.save_pretrained(DATA_DIR)
    print(f" Đã huấn luyện tokenizer và lưu vào '{DATA_DIR}'.")
    print(f" Kích thước từ vựng (vocab_size): {tokenizer.vocab_size}")

    # Tính toán độ dài chuỗi mã hóa lớn nhất trong các cặp Q&A
    max_encoded_len = 0
    for qa_pair in QA_PAIRS:
        encoded_len = len(tokenizer.encode(qa_pair))
        if encoded_len > max_encoded_len:
            max_encoded_len = encoded_len
    
    # Lưu độ dài chuỗi lớn nhất vào file
    with open(MAX_SEQ_LEN_FILE, 'w') as f:
        f.write(str(max_encoded_len))
    print(f" Độ dài chuỗi mã hóa lớn nhất ({max_encoded_len}) đã được lưu vào '{MAX_SEQ_LEN_FILE}'.")

    print("--- Chuẩn bị Dữ liệu Hoàn tất ---")

if __name__ == '__main__':
    prepare_qa_data()