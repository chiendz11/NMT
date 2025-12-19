import random
import os
# Thư viện tách từ tiếng Việt
from pyvi import ViTokenizer 
# Thư viện tách từ tiếng Khmer (Cần cài đặt: pip install khmernltk)
from khmernltk import word_tokenize as tokenize_km

# ============================================================
# CẤU HÌNH (Sửa lại đường dẫn file input của bạn tại đây)
# ============================================================
INPUT_KM = 'ALT.khm-vi.khm'   # Tên file tiếng Khmer gốc
INPUT_VI = 'ALT.khm-vi.vi'   # Tên file tiếng Việt gốc (tương ứng)
OUTPUT_DIR = './data/ALT_Khmer/' # Thư mục lưu kết quả

# Tỉ lệ chia (Giữ nguyên: Dev 1k, Test 1k, còn lại Train)
DEV_SIZE = 1000
TEST_SIZE = 1000

def process():
    # Tạo thư mục output nếu chưa có
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Đã tạo thư mục: {OUTPUT_DIR}")

    print("Đang đọc và tách từ (Khmer - Việt)...")
    
    # Kiểm tra file tồn tại
    if not os.path.exists(INPUT_KM) or not os.path.exists(INPUT_VI):
        print(f"LỖI: Không tìm thấy file input '{INPUT_KM}' hoặc '{INPUT_VI}'")
        return

    with open(INPUT_KM, 'r', encoding='utf-8') as f_km, \
         open(INPUT_VI, 'r', encoding='utf-8') as f_vi:
        
        lines_km = f_km.readlines()
        lines_vi = f_vi.readlines()

    assert len(lines_km) == len(lines_vi), "Lỗi: Số dòng 2 file không bằng nhau!"
    print(f"Tổng số câu: {len(lines_km)}")
    
    # Ghép cặp để shuffle không bị lệch
    pairs = list(zip(lines_km, lines_vi))
    random.shuffle(pairs) # Trộn ngẫu nhiên

    # Tách từ (Tokenize)
    processed_pairs = []
    
    # Biến đếm để theo dõi tiến độ
    count = 0
    total = len(pairs)

    for km, vi in pairs:
        count += 1
        if count % 1000 == 0:
            print(f"Đang xử lý dòng {count}/{total}...", end='\r')

        # 1. Tokenize Khmer
        # khmernltk trả về list, nối lại bằng khoảng trắng
        # Lưu ý: check xem dữ liệu gốc có sạch không, strip() để bỏ xuống dòng thừa
        tok_km = ' '.join(tokenize_km(km.strip()))
        
        # 2. Tokenize Việt
        tok_vi = ViTokenizer.tokenize(vi.strip())
        
        processed_pairs.append((tok_km, tok_vi))

    print("\nĐã tách từ xong. Đang chia tập dữ liệu...")

    # Chia tập
    test_set = processed_pairs[:TEST_SIZE]
    dev_set = processed_pairs[TEST_SIZE : TEST_SIZE + DEV_SIZE]
    train_set = processed_pairs[TEST_SIZE + DEV_SIZE :]

    print(f"--> Train: {len(train_set)} câu")
    print(f"--> Dev:   {len(dev_set)} câu")
    print(f"--> Test:  {len(test_set)} câu")

    # Hàm ghi file
    def write_file(data, name):
        # Lưu ý đuôi file đổi thành .km và .vi
        path_km = f"{OUTPUT_DIR}{name}.km"
        path_vi = f"{OUTPUT_DIR}{name}.vi"
        
        with open(path_km, 'w', encoding='utf-8') as fkm, \
             open(path_vi, 'w', encoding='utf-8') as fvi:
            for k, v in data:
                fkm.write(k + '\n')
                fvi.write(v + '\n')
        print(f"Đã ghi: {path_km} & {path_vi}")

    write_file(train_set, 'train_alt')
    write_file(dev_set, 'dev_alt')
    write_file(test_set, 'test_alt')
    print("Hoàn tất chuẩn bị dữ liệu Khmer-Việt!")

if __name__ == "__main__":
    process()