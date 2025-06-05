import json
import os

class CharacterTokenizer:
    def __init__(self, chars=None):
        """
        - Nếu được truyền vào danh sách ký tự (chars), khởi tạo self.chars = sorted(set(chars)).
        - Nếu không, self.chars = [].
        - Từ self.chars, xây dựng self.char_to_idx, self.idx_to_char, self.vocab_size.
        - Thêm 2 token đặc biệt '<pad>' và '<unk>' qua self.add_token, đồng thời lưu pad_token_id, unk_token_id.
        """
        if chars:
            # sửa: phải gán self.chars (không phải self.char)
            self.chars = sorted(list(set(chars)))
        else:
            self.chars = []

        # Xây dựng mapping từ chars ban đầu (chưa có pad/unk)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        # Đặt token đặc biệt
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'

        # Thêm pad và unk, đồng thời lưu lại chỉ số
        self.pad_token_id = self.add_token(self.pad_token)
        self.unk_token_id = self.add_token(self.unk_token)

    def add_token(self, token):
        """
        - Nếu token chưa nằm trong self.char_to_idx, thêm token vào self.chars, cập nhật  self.char_to_idx, idx_to_char, vocab_size.
        - Trả về chỉ số (int) của token đó (cả trường hợp cũ và mới).
        """
        if token not in self.char_to_idx:
            # thêm token vào cuối self.chars
            self.chars.append(token)
            new_index = len(self.chars) - 1
            self.char_to_idx[token] = new_index
            self.idx_to_char[new_index] = token
            # cập nhật vocab_size
            self.vocab_size = len(self.chars)
            return new_index
        else:
            return self.char_to_idx[token]

    def train(self, text_data):
        """
        Xây dựng từ vựng từ toàn bộ ký tự trong text_data:
        - Tạo self.chars = sorted(set(text_data))
        - Xây dựng self.char_to_idx và idx_to_char từ self.chars
        - Cập nhật self.vocab_size
        - Thêm lại '<pad>' và '<unk>' qua add_token, đồng thời cập nhật pad_token_id, unk_token_id
        """
        # Lấy unique character từ text_data
        self.chars = sorted(list(set(text_data)))

        # Xây dựng lại mapping
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        # Thêm token đặc biệt và lưu id
        self.pad_token_id = self.add_token(self.pad_token)
        self.unk_token_id = self.add_token(self.unk_token)

    def encode(self, text):
        """
        Chuyển một chuỗi text thành list các token ID.
        - Duyệt qua mỗi ký tự trong text:
            + Nếu có trong char_to_idx, lấy id.
            + Nếu không, dùng unk_token_id.
        """
        token_ids = []
        for ch in text:
            if ch in self.char_to_idx:
                token_ids.append(self.char_to_idx[ch])
            else:
                token_ids.append(self.unk_token_id)
        return token_ids

    def decode(self, token_ids):
        """
        Chuyển list các token ID về lại chuỗi ký tự.
        - Duyệt mỗi idx trong token_ids:
            + Nếu idx tồn tại trong idx_to_char, lấy ký tự.
            + Nếu không, nối '<unk>'.
        """
        chars = []
        for idx in token_ids:
            if idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])
            else:
                chars.append(self.unk_token)
        return "".join(chars)

    def save_pretrained(self, save_directory):
        """
        Lưu char_to_idx ra file vocab.json trong save_directory.
        - Nếu thư mục chưa tồn tại, tạo mới.
        - Đảm bảo encoding='utf-8'.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        vocab_path = os.path.join(save_directory, "vocab.json")
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.char_to_idx, f, ensure_ascii=False, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory):
        """
        Đọc file 'vocab.json' từ save_directory, khởi tạo tokenizer tương ứng.
        - Đảm bảo file tồn tại, nếu không, raise FileNotFoundError.
        - Load char_to_idx (dict) từ vocab.json (encoding='utf-8').
        - chars = sorted(char_to_idx.keys()).
        - Gọi cls(chars=chars) để khởi tạo instance.
        - Gán lại pad_token_id, unk_token_id cho tokenizer mới (nếu cần).
        """
        vocab_path = os.path.join(save_directory, "vocab.json")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")

        # Chú ý sửa encoding typo 'uft-8' thành 'utf-8'
        with open(vocab_path, "r", encoding='utf-8') as f:
            char_to_idx = json.load(f)

        # Tạo list chars từ key của char_to_idx
        chars = sorted(char_to_idx.keys())

        # Khởi tạo tokenizer mới với chars
        tokenizer = cls(chars=chars)

        # Nếu muốn chắc chắn, ta có thể gán lại như sau:
        tokenizer.pad_token_id = tokenizer.char_to_idx.get(tokenizer.pad_token)
        tokenizer.unk_token_id = tokenizer.char_to_idx.get(tokenizer.unk_token)

        return tokenizer

# # === Ví dụ thử nghiệm nhanh ===
# if __name__ == "__main__":
#     # Bước 1: đọc dữ liệu từ data.txt (hoặc chuỗi text)
#     with open('/mnt/data/data.txt', 'r', encoding='utf-8') as f:
#         text = f.read()

#     # Bước 2: tạo tokenizer, train trên text
#     tokenizer = CharacterTokenizer()
#     tokenizer.train(text)

#     # In thông tin cơ bản
#     print("Vocab size (sau train):", tokenizer.vocab_size)
#     print("Pad token ID:", tokenizer.pad_token_id, "  Unk token ID:", tokenizer.unk_token_id)

#     # Bước 3: thử encode/decode
#     sample = "Xin chào!"
#     encoded = tokenizer.encode(sample)
#     decoded = tokenizer.decode(encoded)
#     print(f"Original: {sample}")
#     print(f"Encoded IDs: {encoded}")
#     print(f"Decoded back: {decoded}")

#     # Bước 4: lưu tokenizer ra disk
#     save_dir = "/mnt/data/my_tokenizer"
#     tokenizer.save_pretrained(save_dir)

#     # Bước 5: load lại từ disk, kiểm tra vocab_size, pad/unk IDs, encode/decode
#     loaded = CharacterTokenizer.from_pretrained(save_dir)
#     print("Loaded vocab size:", loaded.vocab_size)
#     print("Loaded Pad ID:", loaded.pad_token_id, "  Loaded Unk ID:", loaded.unk_token_id)
#     # Thử decode với ID chưa có
#     print("Decode unknown IDs [999]:", loaded.decode([999]))
