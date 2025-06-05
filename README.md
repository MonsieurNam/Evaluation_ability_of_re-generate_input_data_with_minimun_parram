
# Đánh giá Khả năng Tái tạo Input với Tham số Tối thiểu

Dự án này tập trung vào việc so sánh khả năng của các kiến trúc mô hình ngôn ngữ khác nhau (Transformer, RNN, N-gram, Lookup Table) trong việc **tái tạo chính xác các chuỗi đầu vào đã học**, với mục tiêu tìm ra cấu hình mô hình có **số lượng tham số huấn luyện được tối thiểu** để đạt được độ chính xác tái tạo cao (lý tưởng là 100%).

## Mục lục

1.  [Giới thiệu](#1-giới-thiệu)
2.  [Cấu trúc Dự án](#2-cấu-trúc-dự-án)
3.  [Cách Sử dụng](#3-cách-sử-dụng)
    *   [Giai đoạn 0: Chuẩn bị Dữ liệu](#giai-đoạn-0-chuẩn-bị-dữ-liệu)
    *   [Giai đoạn 1: Đánh giá Baseline Models](#giai-đoạn-1-đánh-giá-baseline-models)
    *   [Giai đoạn 2: Triển khai & Tinh chỉnh RNN](#giai-đoạn-2-triển-khai--tinh-chỉnh-rnn)
    *   [Giai đoạn 3: Triển khai & Tinh chỉnh Transformer (Tiếp theo)](#giai-đoạn-3-triển-khai--tinh-chỉnh-transformer-tiếp-theo)
    *   [Giai đoạn 4: Phân tích & Báo cáo (Cuối cùng)](#giai-đoạn-4-phân-tích--báo-cáo-cuối-cùng)
4.  [Kết quả Sơ bộ](#4-kết-quả-sơ-bộ)
    *   [Mô hình Transformer (Single Block GPT-2)](#mô-hình-transformer-single-block-gpt-2)
    *   [Mô hình RNN (LSTM/GRU)](#mô-hình-rnn-lstm-gru)
5.  [Phân tích Sơ bộ Kết quả](#5-phân-tích-sơ-bộ-kết-quả)



## 1. Giới thiệu

Trong lĩnh vực Mô hình Ngôn ngữ, Transformer đã chứng tỏ sức mạnh vượt trội trong các nhiệm vụ phức tạp. Tuy nhiên, đối với một nhiệm vụ đơn giản như **tái tạo chính xác dữ liệu đã thấy trong quá trình huấn luyện** (hay còn gọi là "ghi nhớ"), liệu kiến trúc phức tạp như Transformer có thực sự hiệu quả về mặt tham số hơn các mô hình đơn giản hơn như RNN, N-gram, hay thậm chí là một bảng tra cứu?

Dự án này khám phá câu hỏi đó bằng cách triển khai và so sánh các mô hình này trên một tập dữ liệu nhỏ gồm các cặp Hỏi-Đáp.

## 2. Cấu trúc Dự án

```
.
├── models/
│   ├── single_block_gpt2.py      # Định nghĩa kiến trúc Transformer (GPT-2 single block)
│   ├── rnn_model.py              # Định nghĩa kiến trúc RNN (Simple LSTM/GRU)
│   ├── lookup_table_model.py     # Định nghĩa Lookup Table Model (baseline)
│   └── n_gram_model.py           # Định nghĩa N-gram Model (baseline)
├── scripts/
│   ├── prepare_data.py           # Chuẩn bị dữ liệu, huấn luyện tokenizer, tính max_seq_len
│   ├── train_rnn.py              # Script huấn luyện mô hình RNN
│   ├── train_transformer.py      # Script huấn luyện mô hình Transformer (sẽ là training.py cũ)
│   ├── inference_rnn.py          # Script suy luận và đánh giá RNN
│   ├── inference_transformer.py  # Script suy luận và đánh giá Transformer (sẽ là inference.py cũ)
│   └── evaluate_baselines.py     # Script đánh giá Lookup Table và N-gram
├── data/
│   ├── data.txt                  # Dữ liệu huấn luyện thô (QA_PAIRS từ prepare_data.py)
│   ├── vocab.json                # Từ vựng của CharacterTokenizer
│   └── max_seq_len.txt           # Độ dài chuỗi lớn nhất trong data.txt (đã mã hóa)
├── result/                       # Thư mục lưu checkpoint của Transformer
│   └── single_block_model_state_dict.pth
├── result_rnn/                   # Thư mục lưu checkpoint của RNN
│   └── rnn_model_state_dict.pth
├── results/                      # Thư mục lưu kết quả đánh giá (CSV)
│   └── baseline_results.csv
├── utils/
│   └── tokenizer.py              # Custom CharacterTokenizer
└── README.md                     # File này
```

## 3. Cách Sử dụng

Để chạy dự án, hãy làm theo các giai đoạn sau:

### Giai đoạn 0: Chuẩn bị Dữ liệu

Bước này chuẩn bị tập dữ liệu, huấn luyện tokenizer và xác định độ dài chuỗi tối đa.

```bash
python scripts/prepare_data.py
```

*   Kết quả: Thư mục `data/` sẽ được tạo chứa `data.txt`, `vocab.json`, và `max_seq_len.txt`.

### Giai đoạn 1: Đánh giá Baseline Models

Đánh giá hiệu suất của mô hình Lookup Table và N-gram, cung cấp các baseline về hiệu quả tham số và khả năng tái tạo.

```bash
python scripts/evaluate_baselines.py
```

*   Kết quả: Kết quả đánh giá sẽ được in ra console và lưu vào `results/baseline_results.csv`.

### Giai đoạn 2: Triển khai & Tinh chỉnh RNN

Huấn luyện và đánh giá mô hình RNN. Mục tiêu là tìm cấu hình RNN nhỏ nhất có thể tái tạo 100% dữ liệu.

1.  **Huấn luyện RNN:**
    Mở `scripts/train_rnn.py` và điều chỉnh các tham số cấu hình mô hình (EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS) trong phần `# --- Cấu hình mô hình RNN ---`. Bắt đầu với các giá trị nhỏ (ví dụ: `EMBEDDING_DIM = 4, HIDDEN_SIZE = 4, NUM_LAYERS = 1`) và tăng dần.

    ```bash
    python scripts/train_rnn.py
    ```

2.  **Suy luận & Đánh giá RNN:**
    **Quan trọng:** Sau khi huấn luyện, hãy mở `scripts/inference_rnn.py` và **đặt các giá trị `EMBEDDING_DIM`, `HIDDEN_SIZE`, `NUM_LAYERS` sao cho chúng khớp chính xác với cấu hình bạn đã dùng để huấn luyện trong `train_rnn.py`**.

    ```bash
    python scripts/inference_rnn.py
    ```

*   Kết quả: Kết quả huấn luyện và suy luận sẽ được in ra console. Bạn sẽ cần ghi lại các số liệu (tham số, Exact Match Rate, thời gian) cho cấu hình RNN tốt nhất.

### Giai đoạn 3: Triển khai & Tinh chỉnh Transformer (Tiếp theo)

(Đây là giai đoạn tiếp theo trong kế hoạch của bạn. Các script tương ứng là `scripts/train_transformer.py` (đổi tên từ `training.py`) và `scripts/inference_transformer.py` (đổi tên từ `inference.py`). Quy trình tương tự như RNN: điều chỉnh cấu hình, huấn luyện, sau đó cập nhật cấu hình trong file inference và đánh giá.)

### Giai đoạn 4: Phân tích & Báo cáo (Cuối cùng)

Tổng hợp tất cả các kết quả từ Lookup Table, N-gram, RNN và Transformer, phân tích và trình bày một báo cáo đầy đủ.

## 4. Kết quả Sơ bộ

Dưới đây là kết quả sơ bộ từ các lần thử nghiệm gần nhất của bạn. Các kết quả này được thực hiện trên CPU và có thể thay đổi tùy thuộc vào phần cứng và số epoch huấn luyện.

### Mô hình Transformer (Single Block GPT-2)

(Kết quả từ `scripts/inference.py` cũ)

```
▶️ Inference sẽ chạy trên device: cpu
Loaded tokenizer from 'results' (vocab_size = 31).
🔢 Đã đọc max_seq_len: 57 (sẽ dùng làm n_positions cho model).
Loaded model state_dict from 'results\single_block_model_state_dict.pth'.
Model có tổng cộng 728 tham số, trong đó 728 tham số trainable.

--- Bắt đầu Kiểm tra Khả năng Tái tạo (Reproduction) ---
Tham số sinh văn bản: temperature=0.01, top_k=1, top_p=0.95

--- Test Case 1/1 ---
  Prompt: "Question: Xin chào"
  Expected full target length: 57 tokens (from 'Question: Xin chào
Answer: FPT University xin chào bạn!.')
  Generating up to 57 tokens...

  🔹 Generated Text:
  -----------------------------------------------------------
Question: Xin chào
Answer: FPT University xin chào bạn!.
  -----------------------------------------------------------
Tái tạo HOÀN TOÀN chính xác!

--- Kiểm tra Khả năng Tái tạo Hoàn tất ---
```

*   **Cấu hình:** (Không hiển thị chi tiết trong output, nhưng model đã đạt 100% Exact Match Rate với 728 tham số.)
*   **Tổng số Tham số:** 728
*   **Exact Match Rate:** 100.00%
*   **Thời gian suy luận trung bình:** (Không có trong output này, nhưng có thể đo trong Giai đoạn 4)

### Mô hình RNN (LSTM/GRU)

(Kết quả từ `scripts/inference_rnn.py` Trial 1: `EMBEDDING_DIM = 4, HIDDEN_SIZE = 4, NUM_LAYERS = 1`)

```
Inference RNN sẽ chạy trên device: cpu
 Loaded tokenizer from 'result_rnn' (vocab_size = 31).
 Đã đọc max_seq_len: 57 (dùng để padding và max_length khi sinh).
 Loaded model state_dict from 'result_rnn\rnn_model_state_dict.pth'.

--- Bắt đầu Kiểm tra Khả năng Tái tạo (RNN Model) ---
Tham số sinh văn bản: temperature=0.01, top_k=1, top_p=0.95

--- Test Case 1/1 ---
  Prompt: "Question: Xin chào"
  Expected full target length: 57 tokens (from 'Question: Xin chào
Answer: FPT University xin chào bạn!.')
  Generating up to 57 tokens...

 Generated Text:
  -----------------------------------------------------------
Question: Xin chàouestion: Xin chào
Answer: FPT Universi
  -----------------------------------------------------------
Tái tạo KHÔNG chính xác (Token Accuracy: 0.00%).
Target:   'Question: Xin chào
Answer: FPT University xin chào bạn!.'
Generated:'Question: Xin chàouestion: Xin chào
Answer: FPT Universi'

--- Tóm tắt Đánh giá RNN Model ---
  Tổng số test cases: 1
  Số lượng tái tạo chính xác hoàn toàn: 0
  Exact Match Rate: 0.00%
  Thời gian suy luận trung bình: 40.338 ms/chuỗi
```

*   **Cấu hình:** `EMBEDDING_DIM = 4, HIDDEN_SIZE = 4, NUM_LAYERS = 1`
*   **Tổng số Tham số:** (Bạn sẽ thấy trong output của `train_rnn.py`, ví dụ khoảng vài trăm. Cần chạy `train_rnn.py` với cấu hình này và ghi lại từ `torchinfo.summary`.)
*   **Exact Match Rate:** 0.00% (cho cấu hình này)
*   **Thời gian suy luận trung bình:** 40.338 ms/chuỗi

## 5. Phân tích Sơ bộ Kết quả

Từ các kết quả sơ bộ, có thể thấy:

*   **Mô hình Transformer** với cấu hình đã huấn luyện (được lưu trong `result/`) đã đạt được khả năng tái tạo **hoàn hảo (100% Exact Match Rate)** cho test case đầu tiên. Điều này cho thấy nó đã học thuộc chuỗi đầu vào rất tốt.
*   **Mô hình RNN** với cấu hình `(EMBEDDING_DIM=4, HIDDEN_SIZE=4, NUM_LAYERS=1)` đã **thất bại hoàn toàn** trong việc tái tạo. Điều này không gây ngạc nhiên vì cấu hình này cực kỳ nhỏ và có thể không đủ năng lực để ghi nhớ các chuỗi dài và đa dạng.

**Bước tiếp theo cho RNN:**  cần tiếp tục thử nghiệm với các cấu hình RNN lớn hơn (tăng `EMBEDDING_DIM` và/hoặc `HIDDEN_SIZE`, và có thể `NUM_LAYERS`) cho đến khi đạt được 100% Exact Match Rate. Sau đó, bạn sẽ so sánh số lượng tham số của cấu hình RNN nhỏ nhất đó với mô hình Transformer để xem mô hình nào hiệu quả hơn về tham số cho nhiệm vụ ghi nhớ này.

---