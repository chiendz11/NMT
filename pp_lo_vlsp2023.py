import os
import random
import re
import urllib.request
import shutil
import unicodedata
import sentencepiece as spm
from tqdm import tqdm
from pyvi import ViTokenizer

# ================= 1. CẤU HÌNH DATASET =================
# Cấu trúc: (path_to_lo, path_to_vi, source_name)
# Bạn hãy thay đổi đường dẫn trỏ đúng tới file của bạn
DATA_SOURCES = [
    # 1. VLSP 2023
    ('./Vietnamese_Lao/VLSP2023/Train/train2023.lo', './Vietnamese_Lao/VLSP2023/Train/train2023.vi', 'VLSP_Train'),
    # 2. Gemini Synthetic
    ('./Vietnamese_Lao/GeminiTranslate/Lo2Vi/gemini.lo', './Vietnamese_Lao/GeminiTranslate/Lo2Vi/gemini.vi', 'Gemini_Lo2Vi'),
    ('./Vietnamese_Lao/GeminiTranslate/Vi2Lo/gemini.lo', './Vietnamese_Lao/GeminiTranslate/Vi2Lo/gemini.vi', 'Gemini_Vi2Lo'),
    # 3. Google Translate Synthetic
    ('./Vietnamese_Lao/GoogleTranslate/Vi2Lo/translate.lo', './Vietnamese_Lao/GoogleTranslate/Vi2Lo/translate.vi', 'Google_Vi2Lo')
]

# ================= 2. KHỞI TẠO THƯ VIỆN =================
try:
    from laonlp.tokenize import word_tokenize as lao_tokenize
except ImportError:
    print("❌ Lỗi: Thiếu 'laonlp'. Chạy: pip install laonlp")
    exit()

try:
    import fasttext
    # Tắt thông báo warning của fasttext
    if hasattr(fasttext, 'FastText'):
        fasttext.FastText.eprint = lambda x: None
except ImportError:
    print("⚠️ Cảnh báo: Thiếu 'fasttext'. Bỏ qua bước lọc ngôn ngữ.")
    fasttext = None

# ================= CLASS PIPELINE XỬ LÝ (CHUẨN) =================
class NMTPipelineLaoViet:
    def __init__(self, data_dir="./data/LO_VI_Final", source_lang="lo", target_lang="vi"):
        self.data_dir = data_dir
        self.src = source_lang
        self.tgt = target_lang
        
        # Tạo cấu trúc thư mục
        self.temp_dir = os.path.join(data_dir, "01_temp")
        self.final_dir = os.path.join(data_dir, "02_final_ready")
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.final_dir, exist_ok=True)

        # Tải model FastText (dùng để lọc rác ngôn ngữ)
        self.lid_model_path = os.path.join(self.data_dir, 'lid.176.ftz')
        if fasttext:
            self._download_lid_model()

    def _download_lid_model(self):
        if not os.path.exists(self.lid_model_path):
            print(f"⬇️ Đang tải model FastText LangID (130MB)...") # File gốc khá nặng, file ftz nhẹ hơn
            try:
                # Link này là bản nén (compressed) ~900KB - 1MB, rất nhanh
                url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
                urllib.request.urlretrieve(url, self.lid_model_path)
            except Exception as e:
                print(f"⚠️ Không tải được model LangID: {e}")

    # ================= BƯỚC 1: MERGE DATASETS =================
    def step_1_merge(self, dataset_list):
        print(f"\n--- [STEP 1] MERGING {len(dataset_list)} DATASETS ---")
        
        out_src = os.path.join(self.temp_dir, f"merged.raw.{self.src}")
        out_tgt = os.path.join(self.temp_dir, f"merged.raw.{self.tgt}")
        
        total_lines = 0
        
        with open(out_src, 'w', encoding='utf-8') as fs_out, \
             open(out_tgt, 'w', encoding='utf-8') as ft_out:
            
            for src_file, tgt_file, name in dataset_list:
                print(f"-> Đọc: {name}...")
                
                if not os.path.exists(src_file) or not os.path.exists(tgt_file):
                    print(f"   ⚠️ File không tồn tại! Bỏ qua {name}.")
                    continue
                
                # Đọc file
                with open(src_file, 'r', encoding='utf-8') as f: lines_src = f.readlines()
                with open(tgt_file, 'r', encoding='utf-8') as f: lines_tgt = f.readlines()

                # Xử lý lệch dòng (Cực kỳ quan trọng)
                if len(lines_src) != len(lines_tgt):
                    print(f"   ❌ CẢNH BÁO: Lệch dòng ({len(lines_src)} vs {len(lines_tgt)}). Cắt theo min.")
                    min_len = min(len(lines_src), len(lines_tgt))
                    lines_src = lines_src[:min_len]
                    lines_tgt = lines_tgt[:min_len]

                added_count = 0
                for s, t in zip(lines_src, lines_tgt):
                    s, t = s.strip(), t.strip()
                    if s and t: # Chỉ lấy khi cả 2 đều có nội dung
                        fs_out.write(s + "\n")
                        ft_out.write(t + "\n")
                        added_count += 1
                
                print(f"   ✅ Đã thêm {added_count} câu.")
                total_lines += added_count
                            
        print(f"✅ GỘP XONG. TỔNG CỘNG: {total_lines} CÂU.")

    # ================= BƯỚC 2: CLEANING (LAO-VIET SPECIFIC) =================
    def step_2_clean(self, min_len=1, max_len=200):
        print(f"\n--- [STEP 2] CLEANING (LAO-VIET) ---")
        
        lid_model = None
        if fasttext and os.path.exists(self.lid_model_path):
            try: lid_model = fasttext.load_model(self.lid_model_path)
            except: pass

        inp_src = os.path.join(self.temp_dir, f"merged.raw.{self.src}")
        inp_tgt = os.path.join(self.temp_dir, f"merged.raw.{self.tgt}")
        out_src = os.path.join(self.temp_dir, f"cleaned.{self.src}")
        out_tgt = os.path.join(self.temp_dir, f"cleaned.{self.tgt}")

        # --- REGEX PATTERNS ---
        html_pattern = re.compile(r'<.*?>') 
        # Giữ lại các dấu câu cơ bản, loại bỏ các ký tự điều khiển lạ
        non_content_pattern = re.compile(r'^[0-9\W_]+$') 
        # Regex kiểm tra ký tự Lào (Unicode Block: 0E80 - 0EFF)
        lao_char_pattern = re.compile(r'[\u0E80-\u0EFF]')

        kept, removed = 0, 0
        seen = set()

        with open(inp_src, 'r', encoding='utf-8') as fs, \
             open(inp_tgt, 'r', encoding='utf-8') as ft, \
             open(out_src, 'w', encoding='utf-8') as os_f, \
             open(out_tgt, 'w', encoding='utf-8') as ot_f:

            for s, t in tqdm(zip(fs, ft), desc="Cleaning"):
                s_orig, t_orig = s.strip(), t.strip()
                
                # 1. CHUẨN HÓA UNICODE (Rất quan trọng cho tiếng Việt)
                s = unicodedata.normalize('NFC', s_orig)
                t = unicodedata.normalize('NFC', t_orig)

                # 2. XÓA HTML & KÝ TỰ RÁC
                s = html_pattern.sub('', s)
                t = html_pattern.sub('', t)
                
                # 3. KIỂM TRA ĐỘ DÀI & NỘI DUNG RỖNG
                if not s or not t: removed += 1; continue
                
                # Nếu câu chỉ toàn số hoặc ký tự đặc biệt -> Xóa
                if non_content_pattern.match(s) or non_content_pattern.match(t):
                    removed += 1; continue

                # 4. KIỂM TRA NGÔN NGỮ (RULE-BASED)
                # Câu nguồn (Lào) bắt buộc phải có ít nhất 1 ký tự Lào
                if not lao_char_pattern.search(s): 
                    removed += 1; continue

                # 5. LỌC THEO ĐỘ DÀI
                len_s = len(s) # Lào: tính theo character
                len_t = len(t.split()) # Việt: tính theo từ (khoảng trắng)

                if len_s < min_len or len_t < min_len: removed += 1; continue
                if len_t > max_len: removed += 1; continue # Chỉ chặn max tiếng Việt để tránh OOM

                # Tỷ lệ độ dài (Length Ratio Filter)
                # Lào thường nhiều ký tự hơn Việt (do viết liền), nhưng không quá 6 lần
                ratio = len_s / (len_t + 1)
                if ratio > 8.0 or ratio < 0.2: removed += 1; continue

                # 6. KHỬ TRÙNG (DEDUPLICATION)
                pair_hash = f"{s}\t{t}"
                if pair_hash in seen: removed += 1; continue
                seen.add(pair_hash)

                # 7. LANGID (FastText - Nếu có)
                if lid_model:
                    try:
                        # Dự đoán ngôn ngữ
                        pred_s = lid_model.predict(s)[0][0]
                        pred_t = lid_model.predict(t)[0][0]
                        
                        # Nếu source bị nhận diện là tiếng Anh hoặc Trung -> Xóa
                        if "__label__en" in pred_s or "__label__zh" in pred_s: 
                            removed += 1; continue
                        # Nếu target bị nhận diện là tiếng Anh -> Xóa (Trừ khi là câu ngắn vay mượn)
                        if "__label__en" in pred_t and len_t > 5: 
                            removed += 1; continue
                    except: pass

                # OK -> Ghi file
                os_f.write(s + "\n")
                ot_f.write(t + "\n")
                kept += 1

        print(f"✅ Clean xong. Giữ: {kept} | Rác: {removed}")

    # ================= BƯỚC 3: SPLIT TRAIN/DEV/TEST =================
    def step_3_split(self, dev_ratio=0.01):
        print(f"\n--- [STEP 3] SPLITTING ---")
        src_path = os.path.join(self.temp_dir, f"cleaned.{self.src}")
        tgt_path = os.path.join(self.temp_dir, f"cleaned.{self.tgt}")
        
        with open(src_path, encoding='utf-8') as fs, open(tgt_path, encoding='utf-8') as ft:
            data = list(zip(fs.readlines(), ft.readlines()))
        
        random.seed(42)
        random.shuffle(data)
        
        total = len(data)
        n_dev = int(total * dev_ratio)
        n_test = n_dev # Test = Dev
        n_train = total - n_dev - n_test
        
        train_set = data[:n_train]
        dev_set = data[n_train : n_train + n_dev]
        test_set = data[n_train + n_dev:]

        def write_file(dataset, mode):
            s_p = os.path.join(self.temp_dir, f"{mode}.{self.src}")
            t_p = os.path.join(self.temp_dir, f"{mode}.{self.tgt}")
            with open(s_p, 'w', encoding='utf-8') as fs, open(t_p, 'w', encoding='utf-8') as ft:
                for s, t in dataset:
                    fs.write(s)
                    ft.write(t)

        write_file(train_set, "train")
        write_file(dev_set, "dev")
        write_file(test_set, "test")
        print(f"-> Train: {len(train_set)} | Dev: {len(dev_set)} | Test: {len(test_set)}")

    # ================= BƯỚC 4: PRE-TOKENIZATION =================
    def step_4_tokenize(self):
        print(f"\n--- [STEP 4] WORD TOKENIZATION (LaoNLP & PyVi) ---")
        modes = ['train', 'dev', 'test']
        
        for mode in modes:
            # 1. Source (Lào) -> Dùng LaoNLP
            inp_s = os.path.join(self.temp_dir, f"{mode}.{self.src}")
            out_s = os.path.join(self.temp_dir, f"{mode}.tok.{self.src}")
            
            with open(inp_s, 'r', encoding='utf-8') as fi, open(out_s, 'w', encoding='utf-8') as fo:
                for line in tqdm(fi, desc=f"Tok LO {mode}", leave=False):
                    words = lao_tokenize(line.strip())
                    fo.write(" ".join(words) + "\n")

            # 2. Target (Việt) -> Dùng PyVi
            inp_t = os.path.join(self.temp_dir, f"{mode}.{self.tgt}")
            out_t = os.path.join(self.temp_dir, f"{mode}.tok.{self.tgt}")
            
            with open(inp_t, 'r', encoding='utf-8') as fi, open(out_t, 'w', encoding='utf-8') as fo:
                for line in tqdm(fi, desc=f"Tok VI {mode}", leave=False):
                    # Pyvi tạo ra từ ghép nối bằng gạch dưới (VD: Hà_Nội)
                    # Điều này tốt cho BPE học từ ghép
                    toks = ViTokenizer.tokenize(line.strip())
                    fo.write(toks + "\n")

    # ================= BƯỚC 5: SEPARATE BPE TRAINING =================
    def step_5_bpe_separate(self, src_vocab=24000, tgt_vocab=32000):
        print(f"\n--- [STEP 5] SEPARATE BPE TRAINING ---")
        
        # --- 1. TRAIN BPE CHO LÀO ---
        # Với 320k câu, 24k vocab giúp model "chắc" hơn, tránh overfit
        print(f"-> Training Source BPE (LO)... Vocab={src_vocab}")
        src_train_file = os.path.join(self.temp_dir, f"train.tok.{self.src}")
        src_model_prefix = os.path.join(self.final_dir, f"spm_{self.src}")
        
        spm.SentencePieceTrainer.train(
            input=src_train_file,
            model_prefix=src_model_prefix,
            vocab_size=src_vocab,
            character_coverage=1.0, # Bắt buộc 1.0 cho Lào/Việt để không mất ký tự
            model_type='bpe',
            num_threads=os.cpu_count(),
            # Các token đặc biệt cần thiết cho NMT
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece='<pad>', unk_piece='<unk>', bos_piece='<s>', eos_piece='</s>'
        )

        # --- 2. TRAIN BPE CHO VIỆT ---
        print(f"-> Training Target BPE (VI)... Vocab={tgt_vocab}")
        tgt_train_file = os.path.join(self.temp_dir, f"train.tok.{self.tgt}")
        tgt_model_prefix = os.path.join(self.final_dir, f"spm_{self.tgt}")
        
        spm.SentencePieceTrainer.train(
            input=tgt_train_file,
            model_prefix=tgt_model_prefix,
            vocab_size=tgt_vocab,
            character_coverage=1.0, 
            model_type='bpe',
            num_threads=os.cpu_count(),
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece='<pad>', unk_piece='<unk>', bos_piece='<s>', eos_piece='</s>'
        )

        # --- 3. APPLY BPE ---
        print(f"-> Applying BPE models to datasets...")
        sp_src = spm.SentencePieceProcessor()
        sp_src.load(f"{src_model_prefix}.model")
        sp_tgt = spm.SentencePieceProcessor()
        sp_tgt.load(f"{tgt_model_prefix}.model")

        for mode in ['train', 'dev', 'test']:
            # Apply LO
            inp_s = os.path.join(self.temp_dir, f"{mode}.tok.{self.src}")
            out_s = os.path.join(self.final_dir, f"{mode}.bpe.{self.src}")
            with open(inp_s, 'r', encoding='utf-8') as fi, open(out_s, 'w', encoding='utf-8') as fo:
                for line in fi:
                    # encode_as_pieces trả về list các subwords
                    pieces = sp_src.encode_as_pieces(line.strip())
                    fo.write(" ".join(pieces) + "\n")

            # Apply VI
            inp_t = os.path.join(self.temp_dir, f"{mode}.tok.{self.tgt}")
            out_t = os.path.join(self.final_dir, f"{mode}.bpe.{self.tgt}")
            with open(inp_t, 'r', encoding='utf-8') as fi, open(out_t, 'w', encoding='utf-8') as fo:
                for line in fi:
                    pieces = sp_tgt.encode_as_pieces(line.strip())
                    fo.write(" ".join(pieces) + "\n")

        print(f"\n✅ SUCCESS! Dữ liệu đã sẵn sàng tại: {self.final_dir}")
        print(f"   - Train files: train.bpe.lo / train.bpe.vi")
        print(f"   - Vocab Source (Lào): {src_vocab}")
        print(f"   - Vocab Target (Việt): {tgt_vocab}")

# ================= MAIN RUN =================
if __name__ == "__main__":
    
    # 1. Khởi tạo
    pipeline = NMTPipelineLaoViet(
        data_dir="./Laos_vlsp", 
        source_lang="lo",
        target_lang="vi"
    )
    
    # 2. Chạy quy trình
    pipeline.step_1_merge(DATA_SOURCES)
    
    # Clean: max_len=150 là đủ cho NMT, dài quá Transformer học không tốt
    pipeline.step_2_clean(min_len=1, max_len=150)
    
    pipeline.step_3_split(dev_ratio=0.005) # ~1500 câu cho dev/test là đủ
    
    pipeline.step_4_tokenize()
    
    # CHỐT THAM SỐ VOCAB TỐI ƯU CHO 320K CÂU
    # Lào: 24k (Đủ cover, tránh loãng)
    # Việt: 32k (Tiêu chuẩn)
    pipeline.step_5_bpe_separate(src_vocab=24000, tgt_vocab=32000)