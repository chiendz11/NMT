import os
import codecs
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE

# === CẤU HÌNH ===
# Số thao tác ghép: Với dataset VLSP ~100k câu, 15000-20000 là hợp lý nhất.
# Quá lớn (32k) sẽ học vẹt từ trọn vẹn, quá nhỏ sẽ nát từ.
NUM_OPS = 32000

DATA_DIR = './data/Laos_tokenized'   # Đầu vào (lấy từ output bước 1)
OUTPUT_DIR = './data/Laos_bpe'       # Đầu ra (file đã bpe để train model)
BPE_CODES_FILE = os.path.join(OUTPUT_DIR, "codes.bpe")

def train_bpe_wrapper(files_to_train, bpe_codes_file, num_operations):
    print(f"\n--- 1. DANG HOC BPE (LEARN BPE) ---")
    print(f"Merge Ops: {num_operations}")
    print(f"Learning from: {files_to_train}")
    
    if not os.path.exists(os.path.dirname(bpe_codes_file)):
        os.makedirs(os.path.dirname(bpe_codes_file))

    temp_concat_file = "temp_train_combined.txt"
    
    try:
        # Gộp file train nguồn và đích để học chung vocab
        with open(temp_concat_file, 'w', encoding='utf-8') as outfile:
            for fname in files_to_train:
                if os.path.exists(fname):
                    with open(fname, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            outfile.write(line)
                else:
                    print(f"CANH BAO: Khong tim thay {fname}")
                    return

        # Học BPE
        with codecs.open(temp_concat_file, encoding='utf-8') as input_files:
            with codecs.open(bpe_codes_file, 'w', encoding='utf-8') as output_bpe_codes:
                learn_bpe(input_files, output_bpe_codes, num_operations)
        
        print(f"-> Da luu BPE codes tai: {bpe_codes_file}")
        
    except Exception as e:
        print(f"LOI KHI HOC BPE: {e}")
    finally:
        if os.path.exists(temp_concat_file):
            os.remove(temp_concat_file)

def apply_bpe_wrapper(bpe_codes_file, input_file, output_file):
    print(f"Dang ap dung BPE: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
    
    if not os.path.exists(input_file):
        print(f"-> BO QUA: Khong tim thay input {input_file}")
        return

    try:
        with codecs.open(bpe_codes_file, encoding='utf-8') as codes:
            bpe = BPE(codes)
            with codecs.open(input_file, encoding='utf-8') as inp, \
                 codecs.open(output_file, 'w', encoding='utf-8') as out:
                for line in inp:
                    out.write(bpe.process_line(line))
    except Exception as e:
        print(f"LOI APPLY BPE: {e}")

if __name__ == "__main__":
    # --- BƯỚC 1: ĐỊNH NGHĨA FILE TRAIN ĐỂ HỌC ---
    # CHỈ HỌC TRÊN TRAIN SET
    train_src_token = os.path.join(DATA_DIR, 'train2023.token.lo')
    train_tgt_token = os.path.join(DATA_DIR, 'train2023.token.vi')
    
    # Học BPE từ cả 2 file train (Shared Vocabulary)
    train_bpe_wrapper([train_src_token, train_tgt_token], BPE_CODES_FILE, NUM_OPS)
    
    # --- BƯỚC 2: ÁP DỤNG CODE ĐÃ HỌC CHO TOÀN BỘ DATASET ---
    datasets = ['train2023', 'dev2023', 'test2023']
    
    print(f"\n--- 2. DANG AP DUNG BPE (APPLY BPE) ---")
    for prefix in datasets:
        for lang in ['lo', 'vi']:
            # File đầu vào (đã tokenized)
            inp_file = os.path.join(DATA_DIR, f"{prefix}.token.{lang}")
            
            # File đầu ra (đã bpe -> dùng cho model)
            out_file = os.path.join(OUTPUT_DIR, f"{prefix}.bpe.{lang}")
            
            apply_bpe_wrapper(BPE_CODES_FILE, inp_file, out_file)

    print("\n=== HOAN THANH ===")
    print(f"File san sang de train model nam tai: {OUTPUT_DIR}")