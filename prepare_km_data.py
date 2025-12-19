import os
import sys

# Ép encoding console để in log tiếng Việt không lỗi
sys.stdout.reconfigure(encoding='utf-8')

# --- KHỐI XỬ LÝ ĐƯỜNG DẪN THƯ VIỆN (CHỐNG LỖI IMPORT) ---
# Thêm đường dẫn site-packages vào hệ thống để chắc chắn Python tìm thấy thư viện
libs_path = r"C:\DEV_APP\AI project\KC4.0_MultilingualNMT\venv_kc4\lib\site-packages"
if libs_path not in sys.path:
    sys.path.append(libs_path)

# 1. IMPORT ĐÚNG TÊN (THEO YÊU CẦU CỦA BẠN)
try:
    import khmernltk
    print(f"✓ Đã load thành công: {khmernltk.__name__}")
except ImportError:
    print("❌ Vẫn chưa import được 'khmernltk'. Hãy kiểm tra lại tên cài đặt.")
    sys.exit(1)

from pyvi import ViTokenizer
from tqdm import tqdm

# ================= CẤU HÌNH =================
# Đường dẫn file gốc (Lưu ý dấu / hoặc \\ cho đúng Windows)
RAW_DIR = "./data/Khmer" 
RAW_KM = os.path.join(RAW_DIR, "OpenSubtitles.km-vi.km")
RAW_VI = os.path.join(RAW_DIR, "OpenSubtitles.km-vi.vi")

# Đường dẫn file đầu ra
OUT_DIR = "./data/Khmer"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

FILES_OUT = {
    "train": (os.path.join(OUT_DIR, "train.km"), os.path.join(OUT_DIR, "train.vi")),
    "dev":   (os.path.join(OUT_DIR, "dev.km"),   os.path.join(OUT_DIR, "dev.vi")),
    "test":  (os.path.join(OUT_DIR, "test.km"),  os.path.join(OUT_DIR, "test.vi")),
}

LIMITS = {
    "train": 542000, 
    "dev":   3000,
    "test":  3000
}

def process_and_split():
    print(f"--- BẮT ĐẦU XỬ LÝ KHMER - VIETNAMESE ---")
    
    if not os.path.exists(RAW_KM) or not os.path.exists(RAW_VI):
        print(f"LỖI: Không tìm thấy file gốc tại {RAW_DIR}")
        return

    # Mở file ghi (Dùng errors='replace' để chống crash khi gặp ký tự lạ)
    f_train_km = open(FILES_OUT["train"][0], 'w', encoding='utf-8', errors='replace')
    f_train_vi = open(FILES_OUT["train"][1], 'w', encoding='utf-8', errors='replace')
    f_dev_km   = open(FILES_OUT["dev"][0],   'w', encoding='utf-8', errors='replace')
    f_dev_vi   = open(FILES_OUT["dev"][1],   'w', encoding='utf-8', errors='replace')
    f_test_km  = open(FILES_OUT["test"][0],  'w', encoding='utf-8', errors='replace')
    f_test_vi  = open(FILES_OUT["test"][1],  'w', encoding='utf-8', errors='replace')

    count = 0
    total_needed = sum(LIMITS.values())
    error_count = 0

    print("Đang đọc và tách từ...")
    
    # Mở file đọc (Dùng errors='ignore' để bỏ qua dòng lỗi encoding nguồn)
    with open(RAW_KM, 'r', encoding='utf-8', errors='ignore') as f_src, \
         open(RAW_VI, 'r', encoding='utf-8', errors='ignore') as f_trg:
        
        for line_km, line_vi in tqdm(zip(f_src, f_trg), total=total_needed):
            try:
                line_km = line_km.strip()
                line_vi = line_vi.strip()

                if not line_km or not line_vi:
                    continue

                # ========================================================
                # 2. SỬA LẠI TÊN GỌI HÀM CHO KHỚP VỚI IMPORT
                # Import là 'khmernltk' thì gọi hàm phải là 'khmernltk.word_tokenize'
                # ========================================================
                seg_km = " ".join(khmernltk.word_tokenize(line_km))

                # Xử lý tiếng Việt
                seg_vi = ViTokenizer.tokenize(line_vi)

                # Ghi file
                if count < LIMITS["train"]:
                    f_train_km.write(seg_km + '\n')
                    f_train_vi.write(seg_vi + '\n')
                elif count < LIMITS["train"] + LIMITS["dev"]:
                    f_dev_km.write(seg_km + '\n')
                    f_dev_vi.write(seg_vi + '\n')
                elif count < LIMITS["train"] + LIMITS["dev"] + LIMITS["test"]:
                    f_test_km.write(seg_km + '\n')
                    f_test_vi.write(seg_vi + '\n')
                else:
                    break 
                
                count += 1
            
            except Exception:
                # Nếu dòng này bị lỗi, bỏ qua và đếm lỗi
                error_count += 1
                continue

    # Đóng file
    f_train_km.close(); f_train_vi.close()
    f_dev_km.close();   f_dev_vi.close()
    f_test_km.close();  f_test_vi.close()

    print("\n--- HOÀN TẤT! ---")
    print(f"Số câu thành công: {count}")
    print(f"Số câu lỗi bỏ qua: {error_count}")
    print(f"Data đã lưu tại: {OUT_DIR}")

if __name__ == "__main__":
    process_and_split()