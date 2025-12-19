import os
import codecs
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE

# ================= CẤU HÌNH CHO KHMER =================
# ALT dataset nhỏ (18k câu), nên để Ops nhỏ thôi
NUM_OPS = 6000

# Thư mục chứa dữ liệu đầu ra từ bước prepare_alt_km.py
DATA_DIR = './data/ALT_Khmer'       # Input
OUTPUT_DIR = './data/ALT_Khmer' # Output

# Định nghĩa tên file codes
BPE_CODE_KM = os.path.join(OUTPUT_DIR, "codes.bpe.km")
BPE_CODE_VI = os.path.join(OUTPUT_DIR, "codes.bpe.vi")

def train_bpe_wrapper(input_file, output_code_file, num_operations):
    print(f"\n--- LEARNING BPE cho: {os.path.basename(input_file)} ---")
    print(f"Ops: {num_operations}")
    
    if not os.path.exists(os.path.dirname(output_code_file)):
        os.makedirs(os.path.dirname(output_code_file))

    try:
        with codecs.open(input_file, encoding='utf-8') as infile:
            with codecs.open(output_code_file, 'w', encoding='utf-8') as outfile:
                learn_bpe(infile, outfile, num_operations)
        print(f"-> Đã lưu code BPE tại: {output_code_file}")
    except Exception as e:
        print(f"LỖI TRAIN BPE: {e}")

def apply_bpe_wrapper(bpe_codes_file, input_file, output_file):
    print(f"Applying BPE: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
    
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

if __name__ == "__main__":
    # Tạo thư mục output nếu chưa có
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- BƯỚC 1: HỌC BPE RIÊNG CHO TỪNG NGÔN NGỮ ---
    # Lưu ý: File đầu vào từ bước trước tên là train_alt.km và train_alt.vi
    train_src_token = os.path.join(DATA_DIR, 'train_alt.km')
    train_tgt_token = os.path.join(DATA_DIR, 'train_alt.vi')
    
    # 1. Học cho Tiếng Khmer (BẮT BUỘC HỌC MỚI)
    train_bpe_wrapper(train_src_token, BPE_CODE_KM, NUM_OPS)
    
    # 2. Học cho Tiếng Việt
    # LƯU Ý QUAN TRỌNG: 
    # Nếu bạn định dùng lại Decoder của model VLSP cũ, bạn KHÔNG NÊN chạy dòng dưới này.
    # Thay vào đó, hãy copy file 'codes.bpe.vi' từ thư mục Laos_bpe sang đây.
    # Nhưng nếu bạn train model Khmer từ đầu (scratch), hãy cứ chạy dòng dưới.
    train_bpe_wrapper(train_tgt_token, BPE_CODE_VI, NUM_OPS)
    
    # --- BƯỚC 2: ÁP DỤNG BPE ---
    # Tên dataset khớp với prepare_alt_km.py
    datasets = ['train_alt', 'dev_alt', 'test_alt']
    
    print(f"\n--- 2. ÁP DỤNG BPE ---")
    for prefix in datasets:
        # 2.1 Áp dụng code Khmer cho file .km
        inp_km = os.path.join(DATA_DIR, f"{prefix}.km")
        out_km = os.path.join(OUTPUT_DIR, f"{prefix}.bpe.km")
        apply_bpe_wrapper(BPE_CODE_KM, inp_km, out_km)
        
        # 2.2 Áp dụng code Việt cho file .vi
        inp_vi = os.path.join(DATA_DIR, f"{prefix}.vi")
        out_vi = os.path.join(OUTPUT_DIR, f"{prefix}.bpe.vi")
        apply_bpe_wrapper(BPE_CODE_VI, inp_vi, out_vi)

    print("\n=== HOÀN THÀNH ===")