import os
import codecs
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE


NUM_OPS = 3000

DATA_DIR = './data/ALT_Laos'   # Input
OUTPUT_DIR = './data/ALT_Laos'       # Output

# Định nghĩa 2 file codes riêng biệt
BPE_CODE_LO = os.path.join(OUTPUT_DIR, "codes.bpe.lo")
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
    # --- BƯỚC 1: HỌC BPE RIÊNG CHO TỪNG NGÔN NGỮ ---
    train_src_token = os.path.join(DATA_DIR, 'train_alt.lo')
    train_tgt_token = os.path.join(DATA_DIR, 'train_alt.vi')
    
    # 1. Học cho Tiếng Lào
    train_bpe_wrapper(train_src_token, BPE_CODE_LO, NUM_OPS)
    
    # 2. Học cho Tiếng Việt
    train_bpe_wrapper(train_tgt_token, BPE_CODE_VI, NUM_OPS)
    
    # --- BƯỚC 2: ÁP DỤNG BPE ---
    datasets = ['train_alt', 'dev_alt', 'test_alt']
    
    print(f"\n--- 2. ÁP DỤNG BPE ---")
    for prefix in datasets:
        # 2.1 Áp dụng code Lào cho file .lo
        inp_lo = os.path.join(DATA_DIR, f"{prefix}.lo")
        out_lo = os.path.join(OUTPUT_DIR, f"{prefix}.bpe.lo")
        apply_bpe_wrapper(BPE_CODE_LO, inp_lo, out_lo)
        
        # 2.2 Áp dụng code Việt cho file .vi
        inp_vi = os.path.join(DATA_DIR, f"{prefix}.vi")
        out_vi = os.path.join(OUTPUT_DIR, f"{prefix}.bpe.vi")
        apply_bpe_wrapper(BPE_CODE_VI, inp_vi, out_vi)

    print("\n=== HOÀN THÀNH ===")