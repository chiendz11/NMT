import os
import pickle
from collections import Counter
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE
import codecs

# ================= CẤU HÌNH =================
DATA_DIR = "./data/Zh"
# Số lượng thao tác gộp (Merge operations). 
# Với data 500k dòng, 32000 là con số tiêu chuẩn.
NUM_OPERATIONS = 32000 

# Định nghĩa các file đầu vào (đã tách từ ở bước trước)
FILES = {
    "train": {"src": "train.zh", "trg": "train.vi"},
    "dev":   {"src": "dev.zh",   "trg": "dev.vi"},
    "test":  {"src": "test.zh",  "trg": "test.vi"}
}

def train_bpe(input_path, code_path, num_ops):
    """Học luật BPE từ file train"""
    print(f"Dataset đang học BPE: {input_path} ...")
    with codecs.open(input_path, encoding='utf-8') as input_file:
        with codecs.open(code_path, 'w', encoding='utf-8') as output_file:
            learn_bpe(input_file, output_file, num_ops)
    print(f"-> Đã lưu codes tại: {code_path}")

def apply_bpe_to_file(bpe_model, input_path, output_path):
    """Áp dụng BPE vào file text"""
    print(f"Đang áp dụng BPE cho: {input_path} -> {output_path}")
    with codecs.open(input_path, encoding='utf-8') as input_file:
        with codecs.open(output_path, 'w', encoding='utf-8') as output_file:
            for line in input_file:
                output_file.write(bpe_model.process_line(line))

def build_vocab(file_path, save_path, min_freq=2):
    """Tạo file vocab .pkl từ file đã BPE"""
    print(f"Đang build vocab từ: {file_path} ...")
    counter = Counter()
    
    with codecs.open(file_path, encoding='utf-8') as f:
        for line in f:
            counter.update(line.strip().split())
    
    # Tạo dictionary: token -> id
    # Các token đặc biệt thường là: <unk>, <pad>, <s>, </s>
    # Tùy code gốc của bạn, nhưng thường torchtext hoặc custom loader sẽ cần 1 dict
    vocab = {
        '<unk>': 0,
        '<pad>': 1,
        '<s>': 2,
        '</s>': 3
    }
    
    # Sắp xếp theo frequency giảm dần
    sorted_words = counter.most_common()
    
    idx = 4 # Bắt đầu sau các token đặc biệt
    for word, freq in sorted_words:
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
            
    # Lưu file pickle
    with open(save_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    print(f"-> Đã lưu Vocab ({len(vocab)} từ) tại: {save_path}")

def main():
    # 1. Đường dẫn file Code BPE
    src_code = os.path.join(DATA_DIR, "codes.bpe.zh")
    trg_code = os.path.join(DATA_DIR, "codes.bpe.vi")
    
    # 2. Học BPE (Chỉ học trên tập Train)
    # Tiếng Trung và Tiếng Việt khác hệ chữ cái hoàn toàn -> Học riêng (Separate BPE)
    train_bpe(os.path.join(DATA_DIR, FILES['train']['src']), src_code, NUM_OPERATIONS)
    train_bpe(os.path.join(DATA_DIR, FILES['train']['trg']), trg_code, NUM_OPERATIONS)
    
    # Load model BPE đã học
    print("Đang load BPE codes...")
    with codecs.open(src_code, encoding='utf-8') as f:
        bpe_zh = BPE(f)
    with codecs.open(trg_code, encoding='utf-8') as f:
        bpe_vi = BPE(f)
        
    # 3. Áp dụng BPE cho tất cả các file (Train, Dev, Test)
    for split, files in FILES.items():
        # Xử lý Source (ZH)
        inp_zh = os.path.join(DATA_DIR, files['src'])
        out_zh = os.path.join(DATA_DIR, f"{split}.bpe.zh") # VD: train.bpe.zh
        apply_bpe_to_file(bpe_zh, inp_zh, out_zh)
        
        # Xử lý Target (VI)
        inp_vi = os.path.join(DATA_DIR, files['trg'])
        out_vi = os.path.join(DATA_DIR, f"{split}.bpe.vi") # VD: train.bpe.vi
        apply_bpe_to_file(bpe_vi, inp_vi, out_vi)

    # 4. Build Vocab (Chỉ từ tập Train đã BPE) và lưu .pkl
    # Đây là file mà model cần để load embedding
    build_vocab(os.path.join(DATA_DIR, "train.bpe.zh"), os.path.join(DATA_DIR, "vocab.bpe.zh.pkl"))
    build_vocab(os.path.join(DATA_DIR, "train.bpe.vi"), os.path.join(DATA_DIR, "vocab.bpe.vi.pkl"))

    print("\n=== HOÀN TẤT TOÀN BỘ QUÁ TRÌNH ===")
    print("File sẵn sàng để train nằm trong folder data/Zh/")

if __name__ == "__main__":
    main()