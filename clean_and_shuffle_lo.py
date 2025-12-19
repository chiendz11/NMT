import random
import os

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
# File đầu vào (là file bạn vừa gộp xong)
INPUT_SRC = 'final_train.lo'
INPUT_TRG = 'final_train.vi'

# File đầu ra (sạch sẽ, dùng để chạy BPE và Train)
OUTPUT_SRC = 'train_dataset.lo'
OUTPUT_TRG = 'train_dataset.vi'

SEED = 42 # Con số may mắn (để kết quả lần nào chạy cũng giống nhau)

def clean_and_shuffle():
    print(f"--- BẮT ĐẦU XỬ LÝ DỮ LIỆU ---")
    
    # 1. Đọc dữ liệu
    print(f"-> Đang đọc file {INPUT_SRC} và {INPUT_TRG}...")
    if not os.path.exists(INPUT_SRC) or not os.path.exists(INPUT_TRG):
        print("LỖI: Không tìm thấy file đầu vào!")
        return

    with open(INPUT_SRC, 'r', encoding='utf-8') as f_src, \
         open(INPUT_TRG, 'r', encoding='utf-8') as f_trg:
        src_lines = f_src.readlines()
        trg_lines = f_trg.readlines()

    # Kiểm tra lệch dòng
    if len(src_lines) != len(trg_lines):
        print(f"CẢNH BÁO: Số dòng bị lệch ({len(src_lines)} vs {len(trg_lines)}).")
        min_len = min(len(src_lines), len(trg_lines))
        src_lines = src_lines[:min_len]
        trg_lines = trg_lines[:min_len]

    original_count = len(src_lines)
    print(f"-> Tổng số dòng ban đầu: {original_count}")

    # 2. Ghép cặp và làm sạch (Deduplicate + Remove Empty)
    print("-> Đang loại bỏ trùng lặp và dòng trống...")
    
    # Dùng set để lọc trùng (tự động loại bỏ các cặp giống hệt nhau)
    unique_pairs = set()
    
    for s, t in zip(src_lines, trg_lines):
        s_clean = s.strip()
        t_clean = t.strip()
        
        # Chỉ lấy nếu cả 2 câu đều có nội dung (không rỗng)
        if s_clean and t_clean:
            unique_pairs.add((s_clean, t_clean))
            
    dedup_count = len(unique_pairs)
    removed_count = original_count - dedup_count
    print(f"-> Đã loại bỏ: {removed_count} dòng (trùng lặp hoặc rỗng).")
    print(f"-> Còn lại: {dedup_count} dòng duy nhất.")

    # 3. Trộn dữ liệu (Shuffle)
    print("-> Đang trộn ngẫu nhiên (Shuffle)...")
    final_data = list(unique_pairs) # Chuyển về list để shuffle
    random.seed(SEED)
    random.shuffle(final_data)

    # 4. Ghi ra file
    print(f"-> Đang ghi ra file {OUTPUT_SRC} và {OUTPUT_TRG}...")
    with open(OUTPUT_SRC, 'w', encoding='utf-8') as f_out_src, \
         open(OUTPUT_TRG, 'w', encoding='utf-8') as f_out_trg:
        
        for s, t in final_data:
            f_out_src.write(s + '\n')
            f_out_trg.write(t + '\n')

    print(f"\n=== HOÀN TẤT ===")
    print(f"File sẵn sàng để chạy BPE: \n 1. {OUTPUT_SRC}\n 2. {OUTPUT_TRG}")

if __name__ == "__main__":
    clean_and_shuffle()