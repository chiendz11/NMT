import os
from subword_nmt.apply_bpe import BPE
import codecs

# CẤU HÌNH ĐƯỜNG DẪN (Kiểm tra kỹ lại xem đúng folder chưa)
CODES_DIR = "./data/Laos_bpe"
DATA_DIR = "./data/ALT_Laos"

FILES = ['train_alt', 'dev_alt', 'test_alt']
LANGS = ['lo', 'vi']

def apply_bpe_for_file(bpe_model, input_file, output_file):
    print(f" -> Processing: {input_file} ...")
    with codecs.open(input_file, encoding='utf-8') as inp, \
         codecs.open(output_file, 'w', encoding='utf-8') as out:
        for line in inp:
            out.write(bpe_model.process_line(line))

def main():
    # 1. Load BPE Codes cho từng ngôn ngữ
    bpe_models = {}
    print(">>> Loading BPE codes...")
    
    # Load Code Lào
    with codecs.open(os.path.join(CODES_DIR, "codes.bpe.lo"), encoding='utf-8') as f:
        bpe_models['lo'] = BPE(f)
        
    # Load Code Việt
    with codecs.open(os.path.join(CODES_DIR, "codes.bpe.vi"), encoding='utf-8') as f:
        bpe_models['vi'] = BPE(f)

    # 2. Chạy vòng lặp apply
    print(">>> Applying BPE...")
    for lang in LANGS:
        current_bpe = bpe_models[lang]
        for fname in FILES:
            inp_path = os.path.join(DATA_DIR, f"{fname}.{lang}")      # Ví dụ: train_alt.lo
            out_path = os.path.join(DATA_DIR, f"{fname}.bpe.{lang}")  # Ví dụ: train_alt.bpe.lo
            
            if os.path.exists(inp_path):
                apply_bpe_for_file(current_bpe, inp_path, out_path)
            else:
                print(f"[WARNING] Không tìm thấy file: {inp_path}")

    print("\n>>> HOÀN TẤT! Hãy mở file .bpe lên kiểm tra lại.")

if __name__ == "__main__":
    main()