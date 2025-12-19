import random
from laonlp import tokenize as tokenize_lo
from pyvi import ViTokenizer # Hoặc dùng thư viện tách từ bạn thích

# Cấu hình
INPUT_LO = 'ALT.lo-vi.lo'  # Tên file gốc của bạn
INPUT_VI = 'ALT.lo-vi.vi'
OUTPUT_DIR = './data/ALT_Laos/' # Thư mục lưu kết quả

# Tỉ lệ chia (18k dòng -> Dev 1k, Test 1k, còn lại Train)
DEV_SIZE = 1000
TEST_SIZE = 1000

def process():
    print("Đang đọc và tách từ...")
    with open(INPUT_LO, 'r', encoding='utf-8') as f_lo, \
         open(INPUT_VI, 'r', encoding='utf-8') as f_vi:
        
        lines_lo = f_lo.readlines()
        lines_vi = f_vi.readlines()

    assert len(lines_lo) == len(lines_vi), "Lỗi: Số dòng 2 file không bằng nhau!"
    
    # Ghép cặp để shuffle không bị lệch
    pairs = list(zip(lines_lo, lines_vi))
    random.shuffle(pairs) # Trộn ngẫu nhiên

    # Tách từ (Tokenize)
    processed_pairs = []
    for lo, vi in pairs:
        # Tokenize Lào
        tok_lo = ' '.join(tokenize_lo.word_tokenize(lo.strip()))
        # Tokenize Việt (dùng pyvi cho nhanh, hoặc thay bằng công cụ bạn đã dùng lúc pre-train)
        tok_vi = ViTokenizer.tokenize(vi.strip())
        processed_pairs.append((tok_lo, tok_vi))

    # Chia tập
    test_set = processed_pairs[:TEST_SIZE]
    dev_set = processed_pairs[TEST_SIZE : TEST_SIZE + DEV_SIZE]
    train_set = processed_pairs[TEST_SIZE + DEV_SIZE :]

    print(f"Train: {len(train_set)}, Dev: {len(dev_set)}, Test: {len(test_set)}")

    # Hàm ghi file
    def write_file(data, name):
        with open(f"{OUTPUT_DIR}{name}.lo", 'w', encoding='utf-8') as flo, \
             open(f"{OUTPUT_DIR}{name}.vi", 'w', encoding='utf-8') as fvi:
            for l, v in data:
                flo.write(l + '\n')
                fvi.write(v + '\n')

    write_file(train_set, 'train_alt')
    write_file(dev_set, 'dev_alt')
    write_file(test_set, 'test_alt')
    print("Hoàn tất Bước 1 & 2!")

if __name__ == "__main__":
    process()