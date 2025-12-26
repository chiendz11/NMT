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
    km_tokenize = lambda x: x.split() 

# --- CẤU HÌNH IMPORT FASTTEXT (LangID) ---
try:
    import fasttext
    if hasattr(fasttext, 'FastText'):
        fasttext.FastText.eprint = lambda x: None
except ImportError:
    print("⚠️ CẢNH BÁO: Thiếu fasttext.")
    fasttext = None

class NMTPipeline:
    def __init__(self, data_dir="data_workspace", source_lang="km", target_lang="vi"):
        self.data_dir = data_dir
        self.src = source_lang
        self.tgt = target_lang
        
        self.temp_dir = os.path.join(data_dir, "01_temp")
        self.final_dir = os.path.join(data_dir, "02_final_ready")
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.final_dir, exist_ok=True)

        self.lid_model_path = os.path.join(self.data_dir, 'lid.176.ftz')
        self._download_lid_model()

    def _download_lid_model(self):
        if not os.path.exists(self.lid_model_path):
            try:
                url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
                urllib.request.urlretrieve(url, self.lid_model_path)
            except: pass

    # ================= BƯỚC 1: MERGE =================
    def step_1_merge(self, datasets):
        print(f"\n--- [STEP 1] MERGING DATA ---")
        out_src = os.path.join(self.temp_dir, f"merged.raw.{self.src}")
        out_tgt = os.path.join(self.temp_dir, f"merged.raw.{self.tgt}")
        
        count = 0
        with open(out_src, 'w', encoding='utf-8') as fs, open(out_tgt, 'w', encoding='utf-8') as ft:
            for path_s, path_t, tag in datasets:
                print(f"-> Processing: {tag}...")
                if not os.path.exists(path_s) or not os.path.exists(path_t): continue
                with open(path_s, 'r', encoding='utf-8', errors='ignore') as f_in_s, \
                     open(path_t, 'r', encoding='utf-8', errors='ignore') as f_in_t:
                    for line_s, line_t in zip(f_in_s, f_in_t):
                        line_s, line_t = line_s.strip(), line_t.strip()
                        if line_s and line_t:
                            fs.write(f"<{tag}> {line_s}\n")
                            ft.write(f"{line_t}\n")
                            count += 1
        print(f"✅ Merge xong. Tổng: {count}")

    # ================= BƯỚC 2: CLEANING =================
    def step_2_clean(self, min_len=1, max_len=400, min_ratio=0.5, max_ratio=2.5):
        print(f"\n--- [STEP 2] ADVANCED CLEANING ---")
        lid_model = None
        if fasttext and os.path.exists(self.lid_model_path):
            try: lid_model = fasttext.load_model(self.lid_model_path)
            except: pass

        inp_src = os.path.join(self.temp_dir, f"merged.raw.{self.src}")
        inp_tgt = os.path.join(self.temp_dir, f"merged.raw.{self.tgt}")
        out_src = os.path.join(self.temp_dir, f"cleaned.{self.src}")
        out_tgt = os.path.join(self.temp_dir, f"cleaned.{self.tgt}")
        
        url_pattern = re.compile(r'http[s]?://\S+')
        kept, removed = 0, 0
        seen = set()

        with open(inp_src, 'r', encoding='utf-8') as fs, open(inp_tgt, 'r', encoding='utf-8') as ft, \
             open(out_src, 'w', encoding='utf-8') as os_f, open(out_tgt, 'w', encoding='utf-8') as ot_f:

            for s, t in tqdm(zip(fs, ft), desc="Filtering"):
                s_orig, t_orig = s.strip(), t.strip()
                
                # Tách tag để check nội dung
                s_content = s_orig
                parts = s_orig.split(' ', 1)
                if len(parts) > 1 and parts[0].startswith('<') and parts[0].endswith('>'):
                    s_content = parts[1]

                # Filter logic
                len_s, len_t = len(s_content), len(t_orig)
                if len_s < min_len or len_t < min_len: removed += 1; continue
                if len_s > max_len or len_t > max_len: removed += 1; continue
                
                ratio = len_s / (len_t + 1e-6)
                if ratio < min_ratio or ratio > max_ratio: removed += 1; continue

                pair_hash = f"{s_content}\t{t_orig}"
                if pair_hash in seen: removed += 1; continue
                seen.add(pair_hash)

                if s_content.lower() == t_orig.lower(): removed += 1; continue
                if url_pattern.search(s_content) or url_pattern.search(t_orig): removed += 1; continue

                if lid_model:
                    try:
                        pred_s = lid_model.predict(s_content)[0][0]
                        pred_t = lid_model.predict(t_orig)[0][0]
                        if f"__label__{self.src}" not in pred_s:
                             if "__label__en" in pred_s or "__label__zh" in pred_s: removed += 1; continue
                        if f"__label__{self.tgt}" not in pred_t:
                             if "__label__en" in pred_t or "__label__zh" in pred_t: removed += 1; continue
                    except: pass

                os_f.write(s_orig + "\n")
                ot_f.write(t_orig + "\n")
                kept += 1
        print(f"✅ Clean xong. Giữ: {kept} | Rác: {removed}")

    # ================= BƯỚC 3: SPLIT =================
    def step_3_split(self, train_ratio=0.99, dev_ratio=0.005):
        print(f"\n--- [STEP 3] SPLITTING ---")
        src_path = os.path.join(self.temp_dir, f"cleaned.{self.src}")
        tgt_path = os.path.join(self.temp_dir, f"cleaned.{self.tgt}")
        with open(src_path, encoding='utf-8') as fs, open(tgt_path, encoding='utf-8') as ft:
            data = list(zip(fs.readlines(), ft.readlines()))
        
        random.seed(42); random.shuffle(data)
        n_train = int(len(data) * train_ratio)
        n_dev = int(len(data) * dev_ratio)
        
        train = data[:n_train]
        dev = data[n_train:n_train+n_dev]
        test = data[n_train+n_dev:]

        def write(ds, mode, rm_tag=False):
            s_p = os.path.join(self.temp_dir, f"{mode}.{self.src}")
            t_p = os.path.join(self.temp_dir, f"{mode}.{self.tgt}")
            with open(s_p, 'w', encoding='utf-8') as fs, open(t_p, 'w', encoding='utf-8') as ft:
                for s, t in ds:
                    s = s.strip()
                    if rm_tag:
                        parts = s.split(' ', 1)
                        if len(parts) > 1 and parts[0].startswith('<') and parts[0].endswith('>'): s = parts[1]
                    fs.write(s + "\n"); ft.write(t.strip() + "\n")

        write(train, "train"); write(dev, "dev"); write(test, "test", rm_tag=True)
        print(f"-> Train: {len(train)} | Dev: {len(dev)} | Test: {len(test)}")

    # ================= BƯỚC 4: TOKENIZE =================
    def step_4_tokenize(self):
        print(f"\n--- [STEP 4] WORD TOKENIZATION ---")
        for mode in ['train', 'dev', 'test']:
            # KM
            inp = os.path.join(self.temp_dir, f"{mode}.{self.src}")
            out = os.path.join(self.temp_dir, f"{mode}.tok.{self.src}")
            with open(inp, 'r', encoding='utf-8') as fi, open(out, 'w', encoding='utf-8') as fo:
                for line in tqdm(fi, desc=f"Tok KM {mode}", leave=False):
                    line = line.strip()
                    tag, content = "", line
                    parts = line.split(' ', 1)
                    if len(parts) > 1 and parts[0].startswith('<') and parts[0].endswith('>'):
                        tag, content = parts[0], parts[1]
                    toks = km_tokenize(content)
                    if isinstance(toks, list): toks = " ".join(toks)
                    fo.write(f"{tag} {toks}".strip() + "\n")
            # VI
            inp = os.path.join(self.temp_dir, f"{mode}.{self.tgt}")
            out = os.path.join(self.temp_dir, f"{mode}.tok.{self.tgt}")
            with open(inp, 'r', encoding='utf-8') as fi, open(out, 'w', encoding='utf-8') as fo:
                for line in tqdm(fi, desc=f"Tok VI {mode}", leave=False):
                    fo.write(ViTokenizer.tokenize(line.strip()) + "\n")

    # ================= BƯỚC 5: SEPARATE BPE =================
    def step_5_bpe_separate(self, src_vocab=28000, tgt_vocab=32000):
        print(f"\n--- [STEP 5] TRAIN & APPLY SEPARATE BPE ---")
        user_defined = ["<opensub>", "<ccaligned>", "<wiki>"]

        # Train KM
        spm.SentencePieceTrainer.train(
            input=os.path.join(self.temp_dir, f"train.tok.{self.src}"),
            model_prefix=os.path.join(self.final_dir, f"spm_{self.src}"),
            vocab_size=src_vocab, character_coverage=1.0, model_type='bpe',
            user_defined_symbols=",".join(user_defined), num_threads=os.cpu_count()
        )
        # Train VI
        spm.SentencePieceTrainer.train(
            input=os.path.join(self.temp_dir, f"train.tok.{self.tgt}"),
            model_prefix=os.path.join(self.final_dir, f"spm_{self.tgt}"),
            vocab_size=tgt_vocab, character_coverage=1.0, model_type='bpe',
            num_threads=os.cpu_count()
        )

        # Apply
        sp_src = spm.SentencePieceProcessor(); sp_src.load(os.path.join(self.final_dir, f"spm_{self.src}.model"))
        sp_tgt = spm.SentencePieceProcessor(); sp_tgt.load(os.path.join(self.final_dir, f"spm_{self.tgt}.model"))

        for mode in ['train', 'dev', 'test']:
            inp_s = os.path.join(self.temp_dir, f"{mode}.tok.{self.src}")
            out_s = os.path.join(self.final_dir, f"{mode}.bpe.{self.src}")
            inp_t = os.path.join(self.temp_dir, f"{mode}.tok.{self.tgt}")
            out_t = os.path.join(self.final_dir, f"{mode}.bpe.{self.tgt}")
            
            with open(inp_s, 'r', encoding='utf-8') as fis, open(out_s, 'w', encoding='utf-8') as fos, \
                 open(inp_t, 'r', encoding='utf-8') as fit, open(out_t, 'w', encoding='utf-8') as fot:
                
                # Apply BPE bình thường, chưa lọc ở đây
                for ls, lt in zip(fis, fit):
                    ps = sp_src.encode_as_pieces(ls.strip())
                    pt = sp_tgt.encode_as_pieces(lt.strip())
                    fos.write(" ".join(ps) + "\n")
                    fot.write(" ".join(pt) + "\n")

    # ================= [MỚI] BƯỚC 6: FINAL CHECK (QUAN TRỌNG) =================
    def step_6_final_check(self):
        """
        Lọc lại lần cuối các file BPE trong thư mục final_ready.
        Loại bỏ các cặp câu mà 1 trong 2 bên bị rỗng (Min length = 0).
        """
        print(f"\n--- [STEP 6] FINAL SAFETY CHECK (Removing empty lines) ---")
        
        modes = ['train', 'dev', 'test']
        for mode in modes:
            src_path = os.path.join(self.final_dir, f"{mode}.bpe.{self.src}")
            tgt_path = os.path.join(self.final_dir, f"{mode}.bpe.{self.tgt}")
            
            # Tạo file tạm
            src_tmp = src_path + ".tmp"
            tgt_tmp = tgt_path + ".tmp"
            
            kept, removed = 0, 0
            
            with open(src_path, 'r', encoding='utf-8') as fs, open(tgt_path, 'r', encoding='utf-8') as ft, \
                 open(src_tmp, 'w', encoding='utf-8') as fso, open(tgt_tmp, 'w', encoding='utf-8') as fto:
                
                for s, t in zip(fs, ft):
                    s, t = s.strip(), t.strip()
                    # Đây là bước quyết định: Nếu sau khi BPE mà rỗng -> Vứt
                    if len(s) == 0 or len(t) == 0:
                        removed += 1
                        continue
                    
                    fso.write(s + "\n")
                    fto.write(t + "\n")
                    kept += 1
            
            # Ghi đè lại file chính
            os.replace(src_tmp, src_path)
            os.replace(tgt_tmp, tgt_path)
            print(f"-> {mode}: Giữ {kept} | Xóa {removed} câu rỗng.")

# ================= MAIN =================
if __name__ == "__main__":
    
    my_datasets = [
        ("OpenSubtitles.km-vi.km", "OpenSubtitles.km-vi.vi", "opensub"),
        ("MultiCCAligned.km-vi.km", "MultiCCAligned.km-vi.vi", "ccaligned"),
    ]
    
    pipeline = NMTPipeline(data_dir="./data/Khmer", source_lang="km", target_lang="vi")
    
    pipeline.step_1_merge(my_datasets)
    pipeline.step_2_clean(min_len=1, max_len=400)
    pipeline.step_3_split()
    pipeline.step_4_tokenize()
    pipeline.step_5_bpe_separate(src_vocab=28000, tgt_vocab=32000)
    
    # --- THÊM DÒNG NÀY ĐỂ FIX LỖI MIN=0 ---
    pipeline.step_6_final_check()