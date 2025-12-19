import os
import jieba
from pyvi import ViTokenizer
from tqdm import tqdm

# ================= CẤU HÌNH =================
DATA_DIR = "./data/Zh"
RAW_ZH = os.path.join(DATA_DIR, "OpenSubtitles.vi-zh_CN.zh_CN")
RAW_VI = os.path.join(DATA_DIR, "OpenSubtitles.vi-zh_CN.vi")

# Đường dẫn file đầu ra
FILES_OUT = {
    "train": (os.path.join(DATA_DIR, "train.zh"), os.path.join(DATA_DIR, "train.vi")),
    "dev":   (os.path.join(DATA_DIR, "dev.zh"),   os.path.join(DATA_DIR, "dev.vi")),
    "test":  (os.path.join(DATA_DIR, "test.zh"),  os.path.join(DATA_DIR, "test.vi")),
}

# Số lượng câu (GPU 8GB thì train 500k là đẹp nhất)
LIMITS = {
    "train": 500000,
    "dev":   3000,
    "test":  3000
}

def process_and_split():
    print(f"--- BẮT ĐẦU XỬ LÝ DỮ LIỆU TẠI: {DATA_DIR} ---")
    
    # Mở file nguồn
    if not os.path.exists(RAW_ZH) or not os.path.exists(RAW_VI):
        print("LỖI: Không tìm thấy file gốc OpenSubtitles trong thư mục data/Zh/")
        return

    # Mở tất cả file để ghi
    f_train_zh = open(FILES_OUT["train"][0], 'w', encoding='utf-8')
    f_train_vi = open(FILES_OUT["train"][1], 'w', encoding='utf-8')
    f_dev_zh   = open(FILES_OUT["dev"][0],   'w', encoding='utf-8')
    f_dev_vi   = open(FILES_OUT["dev"][1],   'w', encoding='utf-8')
    f_test_zh  = open(FILES_OUT["test"][0],  'w', encoding='utf-8')
    f_test_vi  = open(FILES_OUT["test"][1],  'w', encoding='utf-8')

    count = 0
    total_needed = sum(LIMITS.values())
    
    # Đọc song song 2 file
    with open(RAW_ZH, 'r', encoding='utf-8') as f_src, \
         open(RAW_VI, 'r', encoding='utf-8') as f_trg:
        
        for line_zh, line_vi in tqdm(zip(f_src, f_trg), total=total_needed, desc="Đang xử lý"):
            
            line_zh = line_zh.strip()
            line_vi = line_vi.strip()

            # Bỏ qua dòng rỗng
            if not line_zh or not line_vi:
                continue

            # 1. Tách từ Tiếng Trung (Jieba)
            # "我爱你" -> "我 爱 你"
            seg_zh = " ".join(jieba.cut(line_zh))

            # 2. Tách từ Tiếng Việt (Pyvi)
            # "Hà Nội là thủ đô" -> "Hà_Nội là thủ_đô"
            seg_vi = ViTokenizer.tokenize(line_vi)

            # 3. Phân chia vào các file
            if count < LIMITS["train"]:
                f_train_zh.write(seg_zh + '\n')
                f_train_vi.write(seg_vi + '\n')
            elif count < LIMITS["train"] + LIMITS["dev"]:
                f_dev_zh.write(seg_zh + '\n')
                f_dev_vi.write(seg_vi + '\n')
            elif count < LIMITS["train"] + LIMITS["dev"] + LIMITS["test"]:
                f_test_zh.write(seg_zh + '\n')
                f_test_vi.write(seg_vi + '\n')
            else:
                break # Đủ rồi thì dừng, không cần đọc hết 13 triệu dòng
            
            count += 1

    # Đóng file
    f_train_zh.close(); f_train_vi.close()
    f_dev_zh.close();   f_dev_vi.close()
    f_test_zh.close();  f_test_vi.close()

    print("\n--- HOÀN TẤT! ---")
    print(f"Đã tạo train set: {LIMITS['train']} câu")
    print(f"Đã tạo dev set:   {LIMITS['dev']} câu")
    print(f"Đã tạo test set:  {LIMITS['test']} câu")
    print(f"File nằm tại: {DATA_DIR}")

if __name__ == "__main__":
    process_and_split()