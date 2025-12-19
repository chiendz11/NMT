import os
import pickle
from collections import Counter
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE
import codecs

# ================= CẤU HÌNH =================
DATA_DIR = "./data/Khmer"
NUM_OPERATIONS = 32000 

FILES = {
    "train": {"src": "train.km", "trg": "train.vi"},
    "dev":   {"src": "dev.km",   "trg": "dev.vi"},
    "test":  {"src": "test.km",  "trg": "test.vi"}
}

def train_bpe(input_path, code_path, num_ops):
    print(f"Dataset đang học BPE: {input_path} ...")
    with codecs.open(input_path, encoding='utf-8') as input_file:
        with codecs.open(code_path, 'w', encoding='utf-8') as output_file:
            learn_bpe(input_file, output_file, num_ops)
    print(f"-> Đã lưu codes tại: {code_path}")

def apply_bpe_to_file(bpe_model, input_path, output_path):
    print(f"Applying BPE: {input_path} -> {output_path}")
    with codecs.open(input_path, encoding='utf-8') as input_file:
        with codecs.open(output_path, 'w', encoding='utf-8') as output_file:
            for line in input_file:
                output_file.write(bpe_model.process_line(line))

def build_vocab(file_path, save_path, min_freq=2):
    print(f"Building vocab: {file_path} ...")
    counter = Counter()
    with codecs.open(file_path, encoding='utf-8') as f:
        for line in f:
            counter.update(line.strip().split())
    
    vocab = {'<unk>': 0, '<pad>': 1, '<s>': 2, '</s>': 3}
    idx = 4
    for word, freq in counter.most_common():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
            
    with open(save_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"-> Saved vocab ({len(vocab)} words) to: {save_path}")

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Lỗi: Không tìm thấy thư mục {DATA_DIR}. Hãy chạy prepare_km_data.py trước!")
        return

    # 1. Đường dẫn file Code BPE
    src_code = os.path.join(DATA_DIR, "codes.bpe.km")
    trg_code = os.path.join(DATA_DIR, "codes.bpe.vi")
    
    # 2. Học BPE (Train set only)
    train_bpe(os.path.join(DATA_DIR, FILES['train']['src']), src_code, NUM_OPERATIONS)
    train_bpe(os.path.join(DATA_DIR, FILES['train']['trg']), trg_code, NUM_OPERATIONS)
    
    # Load model BPE
    with codecs.open(src_code, encoding='utf-8') as f:
        bpe_km = BPE(f)
    with codecs.open(trg_code, encoding='utf-8') as f:
        bpe_vi = BPE(f)
        
    # 3. Áp dụng BPE
    for split, files in FILES.items():
        # Source (KM)
        inp_km = os.path.join(DATA_DIR, files['src'])
        out_km = os.path.join(DATA_DIR, f"{split}.bpe.km") 
        apply_bpe_to_file(bpe_km, inp_km, out_km)
        
        # Target (VI)
        inp_vi = os.path.join(DATA_DIR, files['trg'])
        out_vi = os.path.join(DATA_DIR, f"{split}.bpe.vi") 
        apply_bpe_to_file(bpe_vi, inp_vi, out_vi)

    # 4. Build Vocab
    build_vocab(os.path.join(DATA_DIR, "train.bpe.km"), os.path.join(DATA_DIR, "vocab.bpe.km.pkl"))
    build_vocab(os.path.join(DATA_DIR, "train.bpe.vi"), os.path.join(DATA_DIR, "vocab.bpe.vi.pkl"))

    print("\n=== SUCCESS ===")

if __name__ == "__main__":
    main()