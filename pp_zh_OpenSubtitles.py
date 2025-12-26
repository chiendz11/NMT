# -*- coding: utf-8 -*-
"""
PREPROCESSING PIPELINE: ZH - VI (1 MILLION SUBSET EDITION)
Phiên bản: Fixed Quantity Split (Train 1M / Dev 3k / Test 3k)
"""

import os
import random
import re
import urllib.request
import shutil
import unicodedata
import sentencepiece as spm
from tqdm import tqdm
from pyvi import ViTokenizer

# ================= 1. KHỞI TẠO THƯ VIỆN =================
try:
    import jieba
    jieba.setLogLevel(20) 
except ImportError:
    print("❌ Lỗi: Thiếu 'jieba'. Chạy: pip install jieba")
    exit()

try:
    from hanziconv import HanziConv
except ImportError:
    print("⚠️ Cảnh báo: Thiếu 'hanziconv'. Tiếng Trung sẽ không được chuẩn hóa.")
    HanziConv = None

try:
    import fasttext
    if hasattr(fasttext, 'FastText'):
        fasttext.FastText.eprint = lambda x: None
except ImportError:
    print("⚠️ Cảnh báo: Thiếu 'fasttext'. Bỏ qua bước lọc ngôn ngữ.")
    fasttext = None

# ================= CLASS PIPELINE XỬ LÝ =================
class NMTPipelineZHVI:
    def __init__(self, data_dir="./data/ZH_VI_Project", source_lang="zh", target_lang="vi"):
        self.data_dir = data_dir
        self.src = source_lang
        self.tgt = target_lang
        
        self.temp_dir = os.path.join(data_dir, "01_temp")
        self.final_dir = os.path.join(data_dir, "02_final_ready")
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.final_dir, exist_ok=True)

        self.lid_model_path = os.path.join(self.data_dir, 'lid.176.ftz')
        if fasttext:
            self._download_lid_model()

    def _download_lid_model(self):
        if not os.path.exists(self.lid_model_path):
            print(f"⬇️ Đang tải model FastText LangID...")
            try:
                url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
                urllib.request.urlretrieve(url, self.lid_model_path)
            except:
                print("⚠️ Không tải được model LangID.")

    # ================= BƯỚC 1: CHUẨN BỊ DỮ LIỆU =================
    def step_1_prep(self, src_file, tgt_file):
        print(f"\n--- [STEP 1] DATA PREPARATION ---")
        out_src = os.path.join(self.temp_dir, f"raw.{self.src}")
        out_tgt = os.path.join(self.temp_dir, f"raw.{self.tgt}")
        
        if not os.path.exists(src_file) or not os.path.exists(tgt_file):
            print(f"❌ LỖI: Không tìm thấy file đầu vào.")
            exit()
            
        print(f"-> Copying input files...")
        shutil.copyfile(src_file, out_src)
        shutil.copyfile(tgt_file, out_tgt)
        print(f"✅ Đã copy xong.")

    # ================= BƯỚC 2: DEEP CLEANING =================
    def step_2_clean(self, min_len=1, max_len=150):
        print(f"\n--- [STEP 2] ADVANCED CLEANING ---")
        
        lid_model = None
        if fasttext and os.path.exists(self.lid_model_path):
            try: lid_model = fasttext.load_model(self.lid_model_path)
            except: pass

        inp_src = os.path.join(self.temp_dir, f"raw.{self.src}")
        inp_tgt = os.path.join(self.temp_dir, f"raw.{self.tgt}")
        out_src = os.path.join(self.temp_dir, f"cleaned.{self.src}")
        out_tgt = os.path.join(self.temp_dir, f"cleaned.{self.tgt}")

        # Patterns
        html_pattern = re.compile(r'<.*?>') 
        bracket_pattern = re.compile(r'[\{\[\(].*?[\}\]\)]')
        music_pattern = re.compile(r'[♪♫♬]') 
        non_content_pattern = re.compile(r'^[0-9\W_]+$')
        zh_char_pattern = re.compile(r'[\u4e00-\u9fff]')
        bad_words = ["sync", "sub", "www", ".com", ".net", "trans", "dịch", "facebook"]

        kept, removed = 0, 0
        seen = set()

        with open(inp_src, 'r', encoding='utf-8') as fs, \
             open(inp_tgt, 'r', encoding='utf-8') as ft, \
             open(out_src, 'w', encoding='utf-8') as os_f, \
             open(out_tgt, 'w', encoding='utf-8') as ot_f:

            for s, t in tqdm(zip(fs, ft), desc="Cleaning"):
                s, t = s.strip(), t.strip()
                
                # 1. Normalize
                s = unicodedata.normalize('NFC', s)
                t = unicodedata.normalize('NFC', t)
                if HanziConv: s = HanziConv.toSimplified(s)

                # 2. Remove Junk
                s = html_pattern.sub('', s); t = html_pattern.sub('', t)
                s = bracket_pattern.sub('', s); t = bracket_pattern.sub('', t)
                s = music_pattern.sub('', s); t = music_pattern.sub('', t)
                
                if not s or not t: removed += 1; continue

                # 3. Content Filter
                if non_content_pattern.match(s) or non_content_pattern.match(t): removed += 1; continue
                if len(s) < 50:
                    if any(w in s.lower() for w in bad_words) or any(w in t.lower() for w in bad_words):
                        removed += 1; continue

                # 4. Check Valid ZH
                if not zh_char_pattern.search(s): removed += 1; continue

                # 5. Length & Ratio
                len_s = len(s); len_t = len(t.split())
                if len_s < min_len or len_t < min_len: removed += 1; continue
                if len_s > max_len or len_t > max_len: removed += 1; continue
                
                ratio = len_s / (len_t + 1)
                if ratio > 4.0 or ratio < 0.2: removed += 1; continue

                # 6. Deduplication
                pair_hash = f"{s}\t{t}"
                if pair_hash in seen: removed += 1; continue
                seen.add(pair_hash)

                # 7. LangID
                if lid_model:
                    try:
                        pred_s = lid_model.predict(s)[0][0]
                        pred_t = lid_model.predict(t)[0][0]
                        if "__label__en" in pred_s or "__label__th" in pred_s: removed += 1; continue
                        if "__label__en" in pred_t: removed += 1; continue
                    except: pass

                os_f.write(s + "\n"); ot_f.write(t + "\n")
                kept += 1

        print(f"✅ Clean xong. Giữ: {kept} | Rác: {removed}")

    # ================= BƯỚC 3: SPLIT DATA (FIXED QUANTITY) =================
    def step_3_split_fixed(self, n_train=1_000_000, n_dev=3_000, n_test=3_000):
        print(f"\n--- [STEP 3] SPLITTING (Fixed: {n_train} Train, {n_dev} Dev, {n_test} Test) ---")
        src_path = os.path.join(self.temp_dir, f"cleaned.{self.src}")
        tgt_path = os.path.join(self.temp_dir, f"cleaned.{self.tgt}")
        
        # Đọc toàn bộ dữ liệu sạch vào RAM (cần khoảng 2-4GB RAM cho 10 triệu câu, máy 16GB RAM OK)
        print("-> Đang đọc dữ liệu sạch...")
        with open(src_path, encoding='utf-8') as fs, open(tgt_path, encoding='utf-8') as ft:
            data = list(zip(fs.readlines(), ft.readlines()))
        
        total_available = len(data)
        total_needed = n_train + n_dev + n_test

        print(f"-> Tổng câu sạch hiện có: {total_available}")
        print(f"-> Tổng câu cần thiết:    {total_needed}")

        # Logic xử lý nếu thiếu dữ liệu
        if total_available < total_needed:
            print("⚠️ CẢNH BÁO: Không đủ dữ liệu như yêu cầu!")
            print("-> Hệ thống sẽ lấy tối đa có thể theo tỷ lệ tương đương.")
            random.shuffle(data)
            n_dev = int(total_available * 0.003) # Giữ Dev nhỏ
            n_test = n_dev
            n_train = total_available - n_dev - n_test
        else:
            # Shuffle ngẫu nhiên
            print("-> Đang trộn ngẫu nhiên...")
            random.seed(42)
            random.shuffle(data)

        # Cắt dữ liệu
        train_set = data[:n_train]
        dev_set = data[n_train : n_train + n_dev]
        test_set = data[n_train + n_dev : n_train + n_dev + n_test]

        # Ghi file
        def write_file(dataset, mode):
            s_p = os.path.join(self.temp_dir, f"{mode}.{self.src}")
            t_p = os.path.join(self.temp_dir, f"{mode}.{self.tgt}")
            with open(s_p, 'w', encoding='utf-8') as fs, open(t_p, 'w', encoding='utf-8') as ft:
                for s, t in dataset:
                    fs.write(s); ft.write(t)

        write_file(train_set, "train")
        write_file(dev_set, "dev")
        write_file(test_set, "test")
        
        print(f"✅ KẾT QUẢ SPLIT:")
        print(f"   - Train: {len(train_set)} câu")
        print(f"   - Dev:   {len(dev_set)} câu (Dùng để Validate)")
        print(f"   - Test:  {len(test_set)} câu (Dùng để test cuối cùng)")

    # ================= BƯỚC 4: TOKENIZE =================
    def step_4_tokenize(self):
        print(f"\n--- [STEP 4] TOKENIZATION ---")
        for mode in ['train', 'dev', 'test']:
            inp_s = os.path.join(self.temp_dir, f"{mode}.{self.src}")
            out_s = os.path.join(self.temp_dir, f"{mode}.tok.{self.src}")
            with open(inp_s, 'r', encoding='utf-8') as fi, open(out_s, 'w', encoding='utf-8') as fo:
                for line in tqdm(fi, desc=f"Tok ZH {mode}", leave=False):
                    fo.write(" ".join(jieba.cut(line.strip())) + "\n")

            inp_t = os.path.join(self.temp_dir, f"{mode}.{self.tgt}")
            out_t = os.path.join(self.temp_dir, f"{mode}.tok.{self.tgt}")
            with open(inp_t, 'r', encoding='utf-8') as fi, open(out_t, 'w', encoding='utf-8') as fo:
                for line in tqdm(fi, desc=f"Tok VI {mode}", leave=False):
                    fo.write(ViTokenizer.tokenize(line.strip()) + "\n")

    # ================= BƯỚC 5: BPE =================
    def step_5_bpe_separate(self, src_vocab=32000, tgt_vocab=32000):
        print(f"\n--- [STEP 5] BPE TRAINING (Trên tập Train 1M) ---")
        
        # Chỉ train BPE trên tập train 1 triệu câu mới
        spm.SentencePieceTrainer.train(
            input=os.path.join(self.temp_dir, f"train.tok.{self.src}"),
            model_prefix=os.path.join(self.final_dir, f"spm_{self.src}"),
            vocab_size=src_vocab, character_coverage=0.9995, model_type='bpe'
        )
        spm.SentencePieceTrainer.train(
            input=os.path.join(self.temp_dir, f"train.tok.{self.tgt}"),
            model_prefix=os.path.join(self.final_dir, f"spm_{self.tgt}"),
            vocab_size=tgt_vocab, character_coverage=1.0, model_type='bpe'
        )

        sp_src = spm.SentencePieceProcessor(); sp_src.load(os.path.join(self.final_dir, f"spm_{self.src}.model"))
        sp_tgt = spm.SentencePieceProcessor(); sp_tgt.load(os.path.join(self.final_dir, f"spm_{self.tgt}.model"))

        for mode in ['train', 'dev', 'test']:
            print(f"-> Applying BPE: {mode}")
            inp_s = os.path.join(self.temp_dir, f"{mode}.tok.{self.src}")
            inp_t = os.path.join(self.temp_dir, f"{mode}.tok.{self.tgt}")
            out_s = os.path.join(self.final_dir, f"{mode}.bpe.{self.src}")
            out_t = os.path.join(self.final_dir, f"{mode}.bpe.{self.tgt}")

            with open(inp_s, 'r', encoding='utf-8') as fis, open(inp_t, 'r', encoding='utf-8') as fit, \
                 open(out_s, 'w', encoding='utf-8') as fos, open(out_t, 'w', encoding='utf-8') as fot:
                for ls, lt in zip(fis, fit):
                    fos.write(" ".join(sp_src.encode_as_pieces(ls.strip())) + "\n")
                    fot.write(" ".join(sp_tgt.encode_as_pieces(lt.strip())) + "\n")

    # ================= BƯỚC 6: FINAL CHECK =================
    def step_6_final_check(self):
        print(f"\n--- [STEP 6] FINAL SAFETY CHECK ---")
        for mode in ['train', 'dev', 'test']:
            src_path = os.path.join(self.final_dir, f"{mode}.bpe.{self.src}")
            tgt_path = os.path.join(self.final_dir, f"{mode}.bpe.{self.tgt}")
            src_tmp, tgt_tmp = src_path + ".tmp", tgt_path + ".tmp"
            
            kept = 0
            with open(src_path, 'r', encoding='utf-8') as fs, open(tgt_path, 'r', encoding='utf-8') as ft, \
                 open(src_tmp, 'w', encoding='utf-8') as fso, open(tgt_tmp, 'w', encoding='utf-8') as fto:
                for s, t in zip(fs, ft):
                    if s.strip() and t.strip():
                        fso.write(s); fto.write(t); kept += 1
            
            os.replace(src_tmp, src_path); os.replace(tgt_tmp, tgt_path)
            print(f"-> {mode}: Sẵn sàng {kept} câu.")

# ================= MAIN =================
if __name__ == "__main__":
    
    # --- CẤU HÌNH INPUT ---
    SRC_FILE = "OpenSubtitles.vi-zh_CN.zh_CN"
    TGT_FILE = "OpenSubtitles.vi-zh_CN.vi"
    WORK_DIR = "./data/Zh"

    pipeline = NMTPipelineZHVI(data_dir=WORK_DIR)
    
    # 1. Prep
    pipeline.step_1_prep(SRC_FILE, TGT_FILE)
    
    # 2. Clean (Chạy trên toàn bộ dữ liệu gốc để đảm bảo chất lượng pool)
    pipeline.step_2_clean(min_len=1, max_len=150) 
    
    # 3. SPLIT THEO SỐ LƯỢNG MONG MUỐN
    # Tại đây ta set cứng: Train 1M, Dev 3k, Test 3k
    pipeline.step_3_split_fixed(n_train=1_000_000, n_dev=3000, n_test=3000)
    
    # 4, 5, 6: Các bước sau chỉ chạy trên tập data đã cắt nhỏ
    pipeline.step_4_tokenize()
    pipeline.step_5_bpe_separate(src_vocab=32000, tgt_vocab=32000)
    pipeline.step_6_final_check()
    
    print("\n🎉 XONG! DỮ LIỆU ĐÃ SẴN SÀNG CHO 8GB VRAM.")