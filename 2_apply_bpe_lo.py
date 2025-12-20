import os
import codecs
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE

# ================= CẤU HÌNH =================
# Vì dùng chung (Shared) nên ta cộng gộp số lượng ops của 2 bên lại
# 16000 (Lo) + 16000 (Vi) = 32000
NUM_OPS = 8000 

DATA_DIR = './data/ALT_Laos'   # Input
OUTPUT_DIR = './data/ALT_Laos' # Output

# Chỉ định nghĩa 1 file code duy nhất
BPE_CODE_FILE = os.path.join(OUTPUT_DIR, "codes.bpe.shared")

# ================= HÀM HỖ TRỢ =================
def train_bpe_wrapper(input_file, output_code_file, num_operations):
    print(f"\n--- LEARNING SHARED BPE từ file gộp: {os.path.basename(input_file)} ---")
    print(f"Ops: {num_operations}")
    
    if not os.path.exists(os.path.dirname(output_code_file)):
        os.makedirs(os.path.dirname(output_code_file))

    try:
        with codecs.open(input_file, encoding='utf-8') as infile:
            with codecs.open(output_code_file, 'w', encoding='utf-8') as outfile:
                learn_bpe(infile, outfile, num_operations)
        print(f"-> Đã lưu code BPE CHUNG tại: {output_code_file}")
    except Exception as e:
        print(f"LỖI TRAIN BPE: {e}")

def apply_bpe_wrapper(bpe_codes_file, input_file, output_file):
    print(f"Applying Shared BPE: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
    
    if not os.path.exists(input_file):
        print(f"-> BỎ QUA: Không tìm thấy {input_file}")
        return

    try:
        with codecs.open(bpe_codes_file, encoding='utf-8') as codes:
            bpe = BPE(codes)
            with codecs.open(input_file, encoding='utf-8') as inp, \
                 codecs.open(output_file, 'w', encoding='utf-8') as out:
                for line in inp:
                    out.write(bpe.process_line(line))
    except Exception as e:
        print(f"LỖI APPLY BPE: {e}")

def concat_files(file_list, output_file):
    """Hàm nối nhiều file lại thành 1 file tạm để train BPE"""
    print(f"Đang gộp file để train BPE chung: {output_file}")
    with codecs.open(output_file, 'w', encoding='utf-8') as outfile:
        for fname in file_list:
            if os.path.exists(fname):
                with codecs.open(fname, encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)
            else:
                print(f"⚠️ Cảnh báo: Không thấy file {fname}")

# ================= MAIN =================
if __name__ == "__main__":
    # File input gốc (đã tokenized)
    train_src_token = os.path.join(DATA_DIR, 'train_alt.lo')
    train_tgt_token = os.path.join(DATA_DIR, 'train_alt.vi')
    
    # File tạm để chứa nội dung gộp
    concat_train_file = os.path.join(DATA_DIR, 'train.concat.tmp')

    # --- BƯỚC 1: HỌC BPE CHUNG (SHARED) ---
    
    # 1.1 Gộp 2 file train lại
    concat_files([train_src_token, train_tgt_token], concat_train_file)
    
    # 1.2 Học BPE trên file gộp
    if os.path.exists(concat_train_file):
        train_bpe_wrapper(concat_train_file, BPE_CODE_FILE, NUM_OPS)
        
        # (Tùy chọn) Xóa file tạm cho đỡ rác
        # os.remove(concat_train_file) 
    else:
        print("❌ Lỗi: Không tạo được file gộp. Dừng chương trình.")
        exit()
    
    # --- BƯỚC 2: ÁP DỤNG BPE CHUNG ---
    datasets = ['train_alt', 'dev_alt', 'test_alt']
    
    print(f"\n--- 2. ÁP DỤNG SHARED BPE CHO TOÀN BỘ DATASET ---")
    
    # Lưu ý: Bây giờ ta dùng chung BPE_CODE_FILE cho cả .lo và .vi
    for prefix in datasets:
        # 2.1 Áp dụng cho file Lào
        inp_lo = os.path.join(DATA_DIR, f"{prefix}.lo")
        out_lo = os.path.join(OUTPUT_DIR, f"{prefix}.bpe.lo")
        apply_bpe_wrapper(BPE_CODE_FILE, inp_lo, out_lo)
        
        # 2.2 Áp dụng cho file Việt
        inp_vi = os.path.join(DATA_DIR, f"{prefix}.vi")
        out_vi = os.path.join(OUTPUT_DIR, f"{prefix}.bpe.vi")
        apply_bpe_wrapper(BPE_CODE_FILE, inp_vi, out_vi)

    print("\n=== HOÀN THÀNH (SHARED BPE MODE) ===")