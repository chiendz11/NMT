import os

# === CẤU HÌNH ĐƯỜNG DẪN ===
# Định nghĩa các cặp file nguồn (Lào) và đích (Việt) cần gộp
# Cấu trúc: (path_to_lo, path_to_vi, source_name)
DATA_SOURCES = [
    # 1. Dữ liệu VLSP Gốc (Chất lượng cao nhất - Gold Standard)
    (
        './Vietnamese_Lao/VLSP2023/Train/train2023.lo', 
        './Vietnamese_Lao/VLSP2023/Train/train2023.vi', 
        'VLSP_Train'
    ),
    
    # 2. Dữ liệu từ Gemini (Synthetic)
    (
        './Vietnamese_Lao/GeminiTranslate/Lo2Vi/gemini.lo', 
        './Vietnamese_Lao/GeminiTranslate/Lo2Vi/gemini.vi', 
        'Gemini_Lo2Vi'
    ),
    (
        './Vietnamese_Lao/GeminiTranslate/Vi2Lo/gemini.lo', 
        './Vietnamese_Lao/GeminiTranslate/Vi2Lo/gemini.vi', 
        'Gemini_Vi2Lo'
    ),

    # 3. Dữ liệu từ Google Translate (Synthetic)
    (
        './Vietnamese_Lao/GoogleTranslate/Vi2Lo/translate.lo', 
        './Vietnamese_Lao/GoogleTranslate/Vi2Lo/translate.vi', 
        'Google_Vi2Lo'
    )
]

# File đầu ra
OUTPUT_LO = 'final_train.lo'
OUTPUT_VI = 'final_train.vi'

def merge_datasets():
    print(f"--- BẮT ĐẦU HỢP NHẤT DỮ LIỆU ---")
    total_lines = 0
    
    # Mở file đầu ra ở chế độ 'write' (ghi mới)
    with open(OUTPUT_LO, 'w', encoding='utf-8') as f_out_lo, \
         open(OUTPUT_VI, 'w', encoding='utf-8') as f_out_vi:
        
        for src_lo, src_vi, name in DATA_SOURCES:
            print(f"Dang doc: {name}...")
            
            # Kiểm tra file tồn tại
            if not os.path.exists(src_lo) or not os.path.exists(src_vi):
                print(f" -> CẢNH BÁO: Bỏ qua {name} vì không tìm thấy file!")
                continue

            # Đọc dữ liệu
            with open(src_lo, 'r', encoding='utf-8') as f_lo, \
                 open(src_vi, 'r', encoding='utf-8') as f_vi:
                
                lines_lo = f_lo.readlines()
                lines_vi = f_vi.readlines()
                
                # KIỂM TRA ĐỘ LỆCH DÒNG (Quan trọng)
                # Dữ liệu song ngữ bắt buộc số dòng phải bằng nhau
                if len(lines_lo) != len(lines_vi):
                    print(f" -> LỖI NGHIÊM TRỌNG: {name} bị lệch dòng!")
                    print(f"    Lào: {len(lines_lo)} vs Việt: {len(lines_vi)}")
                    print(" -> Cách xử lý: Chỉ lấy số dòng tối thiểu chung.")
                    min_len = min(len(lines_lo), len(lines_vi))
                    lines_lo = lines_lo[:min_len]
                    lines_vi = lines_vi[:min_len]
                
                # Ghi vào file tổng
                for l_lo, l_vi in zip(lines_lo, lines_vi):
                    # Strip khoảng trắng thừa và skip dòng rỗng
                    l_lo_clean = l_lo.strip()
                    l_vi_clean = l_vi.strip()
                    
                    if l_lo_clean and l_vi_clean:
                        f_out_lo.write(l_lo_clean + '\n')
                        f_out_vi.write(l_vi_clean + '\n')
                        total_lines += 1
                
                print(f" -> Đã thêm {len(lines_lo)} dòng từ {name}")

    print(f"\n=== HOÀN TẤT ===")
    print(f"Tổng số câu sau khi gộp: {total_lines}")
    print(f"File kết quả: {OUTPUT_LO}, {OUTPUT_VI}")

if __name__ == "__main__":
    merge_datasets()