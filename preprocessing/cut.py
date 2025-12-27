import random
import os
import sys

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
# Đường dẫn file DEV gốc (31k câu)
SRC_INPUT = 'data/Zh/02_final_ready/dev.bpe.zh'
TRG_INPUT = 'data/Zh/02_final_ready/dev.bpe.vi'

# Đường dẫn file DEV NHỎ (dùng để train)
SRC_OUTPUT = 'data/Zh/02_final_ready/dev_small.bpe.zh'
TRG_OUTPUT = 'data/Zh/02_final_ready/dev_small.bpe.vi'

# Số lượng lấy: 2000 - 3000 là chuẩn bài
TARGET_SIZE = 2000 
SEED = 42

def create_subset():
    # Fix lỗi import
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, total=None): return iterable

    print(f"🚀 BẮT ĐẦU TẠO DEV SET {TARGET_SIZE} CÂU...")
    
    if not os.path.exists(SRC_INPUT):
        print("❌ Lỗi: Không tìm thấy file input!")
        return

    # 1. Đếm dòng
    with open(SRC_INPUT, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"   -> Tổng gốc: {total_lines} dòng.")

    if total_lines < TARGET_SIZE:
        print("⚠️ File gốc nhỏ hơn số cần lấy, copy toàn bộ...")
        indices_to_keep = set(range(total_lines))
    else:
        # 2. Random
        print("2. Đang chọn ngẫu nhiên...")
        random.seed(SEED)
        indices_to_keep = set(random.sample(range(total_lines), TARGET_SIZE))

    # 3. Ghi file
    print("3. Đang ghi ra file dev_small...")
    with open(SRC_INPUT, 'r', encoding='utf-8') as src_in, \
         open(TRG_INPUT, 'r', encoding='utf-8') as trg_in, \
         open(SRC_OUTPUT, 'w', encoding='utf-8') as src_out, \
         open(TRG_OUTPUT, 'w', encoding='utf-8') as trg_out:

        iterator = zip(src_in, trg_in)
        for i, (line_src, line_trg) in tqdm(enumerate(iterator), total=total_lines):
            if i in indices_to_keep:
                src_out.write(line_src)
                trg_out.write(line_trg)

    print(f"\n✅ XONG! Hãy sửa config trỏ về file này: dev_small")

if __name__ == "__main__":
    create_subset()