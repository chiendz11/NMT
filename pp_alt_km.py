# File: NMT/pp_alt_km.py
import os
import random
import logging
import argparse
import unicodedata
import sentencepiece as spm
from tqdm import tqdm
from pyvi import ViTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- IMPORT LIBRARY KHMER ---
try:
    from khmernltk import word_tokenize as km_tokenize
    logger.info("✅ Loaded: khmernltk")
except ImportError:
    try:
        from khmer_nltk import word_tokenize as km_tokenize
        logger.info("✅ Loaded: khmer_nltk")
    except ImportError:
        logger.error("❌ Thiếu thư viện Khmer. Hãy chạy: pip install khmer-nltk")
        exit(1)

class KhmerPipeline:
    def __init__(self, args):
        self.args = args
        self.root = os.path.abspath(args.data_root)
        self.raw_dir = os.path.join(self.root, "ALT") 
        
        self.src_path = os.path.join(self.raw_dir, args.src_file)
        self.tgt_path = os.path.join(self.raw_dir, args.tgt_file)

        # Output folder
        self.out_dir = os.path.join(self.root, "ALT_Khmer")
        self.split_dir = os.path.join(self.out_dir, "01_split")
        self.temp_dir = os.path.join(self.out_dir, "02_temp_tok")
        self.final_dir = os.path.join(self.out_dir, "02_final_separate")
        
        for d in [self.split_dir, self.temp_dir, self.final_dir]:
            os.makedirs(d, exist_ok=True)

    def normalize(self, text):
        return unicodedata.normalize('NFC', text.strip())

    def step_1_split(self):
        logger.info(f"--- [1] SPLIT DATA (W/ DEDUPLICATION) ---")
        if not os.path.exists(self.src_path):
            logger.error(f"❌ File not found: {self.src_path}")
            exit(1)

        raw_data = []
        with open(self.src_path, 'r', encoding='utf-8') as fs, open(self.tgt_path, 'r', encoding='utf-8') as ft:
            for s, t in zip(fs, ft):
                s, t = self.normalize(s), self.normalize(t)
                if s and t: raw_data.append((s, t))

        # --- DEDUPLICATION (QUAN TRỌNG CHO LOW-RESOURCE) ---
        original_count = len(raw_data)
        data = list(set(raw_data)) # Khử trùng
        logger.info(f"Original: {original_count} -> Unique: {len(data)} (Dropped {original_count - len(data)})")
        # ---------------------------------------------------

        random.seed(42)
        random.shuffle(data)
        n = len(data)
        splits = {
            'train': data[:int(n*0.9)],
            'dev': data[int(n*0.9):int(n*0.95)],
            'test': data[int(n*0.95):]
        }

        for mode, items in splits.items():
            logger.info(f"-> Writing {mode}: {len(items)}")
            with open(os.path.join(self.split_dir, f"{mode}.{self.args.src}"), 'w', encoding='utf-8') as fs, \
                 open(os.path.join(self.split_dir, f"{mode}.{self.args.tgt}"), 'w', encoding='utf-8') as ft:
                for s, t in items:
                    fs.write(s + '\n')
                    ft.write(t + '\n')

    def step_2_tokenize(self):
        logger.info("--- [2] TOKENIZE ---")
        modes = ['train', 'dev', 'test']
        for mode in modes:
            # Tokenize Khmer
            inp_s = os.path.join(self.split_dir, f"{mode}.{self.args.src}")
            out_s = os.path.join(self.temp_dir, f"{mode}.tok.{self.args.src}")
            with open(inp_s, 'r', encoding='utf-8') as fi, open(out_s, 'w', encoding='utf-8') as fo:
                for line in tqdm(fi, desc=f"Tok KM {mode}"):
                    toks = km_tokenize(line.strip())
                    fo.write(" ".join(toks) if isinstance(toks, list) else line.strip())
                    fo.write("\n")

            # Tokenize Vietnamese
            inp_t = os.path.join(self.split_dir, f"{mode}.{self.args.tgt}")
            out_t = os.path.join(self.temp_dir, f"{mode}.tok.{self.args.tgt}")
            with open(inp_t, 'r', encoding='utf-8') as fi, open(out_t, 'w', encoding='utf-8') as fo:
                for line in tqdm(fi, desc=f"Tok VI {mode}"):
                    fo.write(ViTokenizer.tokenize(line.strip()) + "\n")

    def step_3_clean(self):
        logger.info("--- [3] CLEAN TRAIN ---")
        inp_s = os.path.join(self.temp_dir, f"train.tok.{self.args.src}")
        inp_t = os.path.join(self.temp_dir, f"train.tok.{self.args.tgt}")
        out_s = os.path.join(self.temp_dir, f"train.clean.{self.args.src}")
        out_t = os.path.join(self.temp_dir, f"train.clean.{self.args.tgt}")

        kept = 0
        with open(inp_s, 'r', encoding='utf-8') as fs, open(inp_t, 'r', encoding='utf-8') as ft, \
             open(out_s, 'w', encoding='utf-8') as os_f, open(out_t, 'w', encoding='utf-8') as ot_f:
            for s, t in zip(fs, ft):
                s, t = s.strip(), t.strip()
                ls, lt = len(s.split()), len(t.split())
                # Lọc câu quá dài hoặc quá ngắn hoặc tỉ lệ độ dài chênh lệch quá lớn
                if ls < 1 or lt < 1 or ls > 400 or lt > 400 or ls/lt > 4.0 or lt/ls > 4.0:
                    continue
                os_f.write(s + '\n')
                ot_f.write(t + '\n')
                kept += 1
        logger.info(f"-> Kept for training: {kept}")

    def step_4_separate_bpe(self):
        logger.info(f"--- [4] SEPARATE BPE (Vocab={self.args.vocab_size}) ---")
        
        train_src = os.path.join(self.temp_dir, f"train.clean.{self.args.src}")
        train_tgt = os.path.join(self.temp_dir, f"train.clean.{self.args.tgt}")
        
        prefix_src = os.path.join(self.final_dir, f"spm_{self.args.src}")
        prefix_tgt = os.path.join(self.final_dir, f"spm_{self.args.tgt}")

        # Train Source (KHMER)
        logger.info(f"Training BPE for {self.args.src}...")
        spm.SentencePieceTrainer.train(
            input=train_src, model_prefix=prefix_src, vocab_size=self.args.vocab_size, 
            model_type='bpe', character_coverage=1.0 # Khmer lấy 100% ký tự
        )

        # Train Target (VIETNAMESE)
        logger.info(f"Training BPE for {self.args.tgt}...")
        spm.SentencePieceTrainer.train(
            input=train_tgt, model_prefix=prefix_tgt, vocab_size=self.args.vocab_size, 
            model_type='bpe', character_coverage=1.0
        )

        # Apply
        sp_s = spm.SentencePieceProcessor(model_file=f"{prefix_src}.model")
        sp_t = spm.SentencePieceProcessor(model_file=f"{prefix_tgt}.model")

        for mode in ['train', 'dev', 'test']:
            prefix = "train.clean" if mode == 'train' else f"{mode}.tok"
            # Apply Src
            with open(os.path.join(self.temp_dir, f"{prefix}.{self.args.src}"), 'r', encoding='utf-8') as fi, \
                 open(os.path.join(self.final_dir, f"{mode}.bpe.{self.args.src}"), 'w', encoding='utf-8') as fo:
                for line in fi: fo.write(" ".join(sp_s.encode_as_pieces(line.strip())) + "\n")
            # Apply Tgt
            with open(os.path.join(self.temp_dir, f"{prefix}.{self.args.tgt}"), 'r', encoding='utf-8') as fi, \
                 open(os.path.join(self.final_dir, f"{mode}.bpe.{self.args.tgt}"), 'w', encoding='utf-8') as fo:
                for line in fi: fo.write(" ".join(sp_t.encode_as_pieces(line.strip())) + "\n")
        
        logger.info(f"✅ DONE. Data saved to: {self.final_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--src_file", default="ALT.khm-vi.khm")
    parser.add_argument("--tgt_file", default="ALT.khm-vi.vi")
    parser.add_argument("--src", default="km")
    parser.add_argument("--tgt", default="vi")
    # ĐÃ ĐỔI MẶC ĐỊNH THÀNH 4000
    parser.add_argument("--vocab_size", type=int, default=4000)
    args = parser.parse_args()

    pipeline = KhmerPipeline(args)
    pipeline.step_1_split()
    pipeline.step_2_tokenize()
    pipeline.step_3_clean()
    pipeline.step_4_separate_bpe()