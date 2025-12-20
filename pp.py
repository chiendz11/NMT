import os
import random
import re
import urllib.request
import sentencepiece as spm
from tqdm import tqdm
from pyvi import ViTokenizer

# --- CẤU HÌNH IMPORT THƯ VIỆN KHMER ---
try:
    from khmer_nltk import word_tokenize as km_tokenize
except ImportError:
    print("⚠️ CẢNH BÁO: Chưa cài 'khmer-nltk'.")
    print("   Chạy: pip install khmer-nltk")
    # Fallback function: tách theo khoảng trắng (chất lượng kém hơn)
    km_tokenize = lambda x: x.split() 

# --- CẤU HÌNH IMPORT FASTTEXT (LangID) ---
try:
    import fasttext
    # Tắt thông báo warning của fasttext
    fasttext.FastText.eprint = lambda x: None
except ImportError:
    print("⚠️ CẢNH BÁO: Chưa cài 'fasttext'. Tính năng lọc ngôn ngữ sẽ bị bỏ qua.")
    fasttext = None

class NMTPipeline:
    def __init__(self, data_dir="data_workspace", source_lang="km", target_lang="vi"):
        self.data_dir = data_dir
        self.src = source_lang
        self.tgt = target_lang
        
        # Setup thư mục
        self.temp_dir = os.path.join(data_dir, "01_temp")
        self.final_dir = os.path.join(data_dir, "02_final_ready")
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.final_dir, exist_ok=True)

        # Download model nhận diện ngôn ngữ nếu chưa có
        self.lid_model_path = os.path.join(self.data_dir, 'lid.176.ftz')
        self._download_lid_model()

    def _download_lid_model(self):
        """Tải model nhận diện ngôn ngữ của Facebook (nhẹ ~1MB)"""
        if not os.path.exists(self.lid_model_path):
            print(f"⬇️ Đang tải model LangID (lid.176.ftz)...")
            url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
            urllib.request.urlretrieve(url, self.lid_model_path)

    # ================= BƯỚC 1: MERGE DỮ LIỆU =================
    def step_1_merge(self, datasets):
        print(f"\n--- [STEP 1] MERGING DATA ---")
        out_src = os.path.join(self.temp_dir, f"merged.raw.{self.src}")
        out_tgt = os.path.join(self.temp_dir, f"merged.raw.{self.tgt}")
        
        count = 0
        with open(out_src, 'w', encoding='utf-8') as fs, \
             open(out_tgt, 'w', encoding='utf-8') as ft:
            
            for path_s, path_t, tag in datasets:
                print(f"-> Processing: {tag} ({os.path.basename(path_s)})...")
                if not os.path.exists(path_s) or not os.path.exists(path_t):
                    print(f"   ❌ File not found: {path_s}. Skip.")
                    continue

                with open(path_s, 'r', encoding='utf-8', errors='ignore') as f_in_s, \
                     open(path_t, 'r', encoding='utf-8', errors='ignore') as f_in_t:
                    
                    for line_s, line_t in zip(f_in_s, f_in_t):
                        line_s, line_t = line_s.strip(), line_t.strip()
                        if line_s and line_t:
                            # Thêm tag vào đầu nguồn để model biết nguồn gốc dữ liệu
                            fs.write(f"<{tag}> {line_s}\n")
                            ft.write(f"{line_t}\n")
                            count += 1
        print(f"✅ Merge xong. Tổng số câu: {count}")
        return out_src, out_tgt

    # ================= BƯỚC 2: ADVANCED CLEANING (BEST PRACTICE) =================
    def step_2_clean(self, min_len=1, max_len=400, min_ratio=0.5, max_ratio=2.5):
        print(f"\n--- [STEP 2] ADVANCED CLEANING (FILTERING) ---")
        
        # Load FastText model
        lid_model = None
        if fasttext and os.path.exists(self.lid_model_path):
            lid_model = fasttext.load_model(self.lid_model_path)
            print("-> Đã load LangID Model (FastText).")
        else:
            print("-> ⚠️ Không dùng LangID filter (thiếu thư viện/model).")

        inp_src = os.path.join(self.temp_dir, f"merged.raw.{self.src}")
        inp_tgt = os.path.join(self.temp_dir, f"merged.raw.{self.tgt}")
        out_src = os.path.join(self.temp_dir, f"cleaned.{self.src}")
        out_tgt = os.path.join(self.temp_dir, f"cleaned.{self.tgt}")

        # Regex patterns
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        email_pattern = re.compile(r'\S+@\S+')
        html_pattern = re.compile(r'<.*?>') # Cẩn thận với thẻ <tag> của mình

        kept, removed = 0, 0
        seen = set()

        with open(inp_src, 'r', encoding='utf-8') as fs, \
             open(inp_tgt, 'r', encoding='utf-8') as ft, \
             open(out_src, 'w', encoding='utf-8') as os_f, \
             open(out_tgt, 'w', encoding='utf-8') as ot_f:

            for s, t in tqdm(zip(fs, ft), desc="Filtering"):
                s_orig, t_orig = s.strip(), t.strip()

                # --- 1. Tách thẻ <tag> ra để check nội dung ---
                # Giả định format: "<tag> content"
                s_content = s_orig
                parts = s_orig.split(' ', 1)
                if len(parts) > 1 and parts[0].startswith('<') and parts[0].endswith('>'):
                    s_content = parts[1] # Chỉ lấy phần nội dung

                # --- 2. Basic Length & Ratio ---
                len_s, len_t = len(s_content), len(t_orig)
                if len_s < min_len or len_t < min_len: removed += 1; continue
                if len_s > max_len or len_t > max_len: removed += 1; continue
                
                ratio = len_s / (len_t + 1e-6)
                if ratio < min_ratio or ratio > max_ratio: removed += 1; continue

                # --- 3. Deduplication ---
                pair_hash = f"{s_content}\t{t_orig}"
                if pair_hash in seen: removed += 1; continue
                seen.add(pair_hash)

                # --- 4. Content Heuristics ---
                # Nếu nguồn và đích giống hệt nhau (thường là tên riêng, nhiễu) -> Bỏ
                if s_content.lower() == t_orig.lower(): removed += 1; continue
                
                # Chứa URL hoặc Email -> Bỏ
                if url_pattern.search(s_content) or url_pattern.search(t_orig): removed += 1; continue
                # if email_pattern.search(s_content) or email_pattern.search(t_orig): removed += 1; continue

                # --- 5. Language Identification (Quan trọng) ---
                if lid_model:
                    try:
                        # Check Source (phải là source_lang hoặc unknown, không được là tiếng Anh/Trung...)
                        pred_s = lid_model.predict(s_content)[0][0]
                        # Check Target
                        pred_t = lid_model.predict(t_orig)[0][0]

                        # Logic lọc: 
                        # Nếu source KHÔNG PHẢI km (ví dụ là en, zh, th) -> Bỏ
                        # Lưu ý: __label__km
                        if f"__label__{self.src}" not in pred_s:
                             # Cho phép 'unk' (unknown) đi qua vì câu ngắn AI hay đoán sai
                            if "__label__en" in pred_s or "__label__zh" in pred_s or "__label__th" in pred_s:
                                removed += 1; continue
                        
                        # Nếu target KHÔNG PHẢI vi -> Bỏ
                        if f"__label__{self.tgt}" not in pred_t:
                            if "__label__en" in pred_t or "__label__zh" in pred_t:
                                removed += 1; continue
                    except:
                        pass # Lỗi dự đoán thì cho qua

                # --- OK: Ghi file ---
                os_f.write(s_orig + "\n") # Ghi cả tag
                ot_f.write(t_orig + "\n")
                kept += 1

        print(f"✅ Clean xong. Giữ: {kept}, Loại bỏ: {removed} (Tỷ lệ rác: {removed/(kept+removed):.2%})")

    # ================= BƯỚC 3: SPLIT =================
    def step_3_split(self, train_ratio=0.99, dev_ratio=0.005):
        print(f"\n--- [STEP 3] SPLITTING DATA ---")
        src_path = os.path.join(self.temp_dir, f"cleaned.{self.src}")
        tgt_path = os.path.join(self.temp_dir, f"cleaned.{self.tgt}")
        
        # Đọc dữ liệu (nếu RAM < 8GB và dữ liệu > 2 triệu câu, nên dùng thư viện khác)
        # Với 320k câu thì vô tư
        with open(src_path, encoding='utf-8') as fs, open(tgt_path, encoding='utf-8') as ft:
            data = list(zip(fs.readlines(), ft.readlines()))
        
        random.seed(42) # Cố định seed để tái lập kết quả
        random.shuffle(data)
        
        total = len(data)
        n_train = int(total * train_ratio)
        n_dev = int(total * dev_ratio)
        
        train_set = data[:n_train]
        dev_set = data[n_train : n_train + n_dev]
        test_set = data[n_train + n_dev:]
        
        def write_split(dataset, name, remove_tag=False):
            s_out = os.path.join(self.temp_dir, f"{name}.{self.src}")
            t_out = os.path.join(self.temp_dir, f"{name}.{self.tgt}")
            with open(s_out, 'w', encoding='utf-8') as fs, open(t_out, 'w', encoding='utf-8') as ft:
                for s, t in dataset:
                    s = s.strip()
                    if remove_tag:
                        # Xóa tag: <opensub> xin chào -> xin chào
                        parts = s.split(' ', 1)
                        if len(parts) > 1 and parts[0].startswith('<') and parts[0].endswith('>'):
                            s = parts[1]
                    fs.write(s + '\n')
                    ft.write(t.strip() + '\n')

        print(f"-> Train: {len(train_set)} | Dev: {len(dev_set)} | Test: {len(test_set)}")
        write_split(train_set, "train")
        write_split(dev_set, "dev")
        write_split(test_set, "test", remove_tag=True) # Test phải sạch tag để đánh giá BLEU

    # ================= BƯỚC 4: WORD TOKENIZATION =================
    def step_4_tokenize(self):
        print(f"\n--- [STEP 4] WORD TOKENIZATION ---")
        modes = ['train', 'dev', 'test']
        
        for mode in modes:
            print(f"-> Tokenizing {mode} set...")
            
            # --- SOURCE (KHMER) ---
            inp_s = os.path.join(self.temp_dir, f"{mode}.{self.src}")
            out_s = os.path.join(self.temp_dir, f"{mode}.tok.{self.src}")
            
            with open(inp_s, 'r', encoding='utf-8') as fi, open(out_s, 'w', encoding='utf-8') as fo:
                for line in tqdm(fi, desc=f"{self.src.upper()} {mode}"):
                    line = line.strip()
                    tag = ""
                    content = line
                    
                    # Tách tag (nếu có)
                    parts = line.split(' ', 1)
                    if len(parts) > 1 and parts[0].startswith('<') and parts[0].endswith('>'):
                        tag = parts[0]
                        content = parts[1]
                    
                    # Tokenize
                    toks = km_tokenize(content)
                    if isinstance(toks, list): toks = " ".join(toks)
                    
                    fo.write(f"{tag} {toks}".strip() + '\n')

            # --- TARGET (VIETNAMESE) ---
            inp_t = os.path.join(self.temp_dir, f"{mode}.{self.tgt}")
            out_t = os.path.join(self.temp_dir, f"{mode}.tok.{self.tgt}")
            
            with open(inp_t, 'r', encoding='utf-8') as fi, open(out_t, 'w', encoding='utf-8') as fo:
                for line in tqdm(fi, desc=f"{self.tgt.upper()} {mode}"):
                    # Pyvi xử lý tốt
                    toks = ViTokenizer.tokenize(line.strip())
                    fo.write(toks + '\n')

    # ================= BƯỚC 5: SHARED BPE =================
    def step_5_bpe(self, vocab_size=32000):
        print(f"\n--- [STEP 5] TRAIN & APPLY SHARED BPE ---")
        
        # 1. Chuẩn bị file train gộp (Concat) để học Vocab chung
        # Lưu ý: Nên gộp cả Dev vào để học vocab cho đủ, tránh OOV
        train_s = os.path.join(self.temp_dir, f"train.tok.{self.src}")
        train_t = os.path.join(self.temp_dir, f"train.tok.{self.tgt}")
        dev_s = os.path.join(self.temp_dir, f"dev.tok.{self.src}")
        dev_t = os.path.join(self.temp_dir, f"dev.tok.{self.tgt}")
        
        concat_file = os.path.join(self.temp_dir, "vocab_learner.concat")
        
        print("-> Gộp dữ liệu để học BPE...")
        with open(concat_file, 'w', encoding='utf-8') as fo:
            for fname in [train_s, train_t, dev_s, dev_t]:
                with open(fname, 'r', encoding='utf-8') as fi:
                    fo.write(fi.read())

        # 2. Train Model SentencePiece
        model_prefix = os.path.join(self.final_dir, "spm_shared")
        
        # Các token đặc biệt do người dùng định nghĩa
        user_defined = ["<opensub>", "<ccaligned>", "<wiki>"]
        
        print(f"-> Training SentencePiece (Vocab={vocab_size})...")
        # Sử dụng API trực tiếp
        spm.SentencePieceTrainer.train(
            input=concat_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=1.0, # BẮT BUỘC CHO TIẾNG VIỆT/KHMER/LÀO
            model_type='bpe',
            user_defined_symbols=",".join(user_defined),
            num_threads=os.cpu_count() # Dùng full CPU
        )

        # 3. Apply
        sp = spm.SentencePieceProcessor()
        sp.load(f"{model_prefix}.model")
        
        modes = ['train', 'dev', 'test']
        for mode in modes:
            for lang in [self.src, self.tgt]:
                inp_file = os.path.join(self.temp_dir, f"{mode}.tok.{lang}")
                out_file = os.path.join(self.final_dir, f"{mode}.bpe.{lang}")
                
                print(f"-> Applying BPE: {mode}.{lang}")
                with open(inp_file, 'r', encoding='utf-8') as fi, open(out_file, 'w', encoding='utf-8') as fo:
                    for line in fi:
                        pieces = sp.encode_as_pieces(line.strip())
                        fo.write(" ".join(pieces) + "\n")
        
        print(f"\n🎉 SUCCESS! Dữ liệu đã sẵn sàng tại: {self.final_dir}")
        print(f"   - Vocab model: {model_prefix}.model")
        print(f"   - Vocab file:  {model_prefix}.vocab")

# ================= MAIN =================
if __name__ == "__main__":
    
    # 1. Định nghĩa dữ liệu đầu vào
    # Tuple: (Path nguồn, Path đích, Tag nhận dạng)
    my_datasets = [
        # Ví dụ đường dẫn, hãy sửa lại cho đúng máy của bạn
        ("OpenSubtitles.km-vi.km", "OpenSubtitles.km-vi.vi", "opensub"),
        ("MultiCCAligned.km-vi.km", "MultiCCAligned.km-vi.vi", "ccaligned"),
    ]
    
    # 2. Khởi tạo Pipeline
    pipeline = NMTPipeline(
        data_dir="./data/Khmer", # Nơi chứa output
        source_lang="km", 
        target_lang="vi"
    )
    
    # 3. Chạy từng bước
    # Bước 1: Gộp
    pipeline.step_1_merge(my_datasets)
    
    # Bước 2: Làm sạch (Deep Clean)
    # min_len=1 (giữ từ đơn như "ừ", "dạ"), max_len=400 (tránh câu quá dài gây OOM)
    pipeline.step_2_clean(min_len=1, max_len=400, min_ratio=0.5, max_ratio=2.5)
    
    # Bước 3: Chia tập (Train 99%, Dev 0.5%, Test 0.5% vì dữ liệu lớn)
    pipeline.step_3_split(train_ratio=0.99, dev_ratio=0.005)
    
    # Bước 4: Tách từ
    pipeline.step_4_tokenize()
    
    # Bước 5: BPE
    # Với 320k câu, vocab=32000 là chuẩn
    pipeline.step_5_bpe(vocab_size=32000)