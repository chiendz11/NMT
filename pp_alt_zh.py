# File: NMT/pp_alt_zh.py
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

try:
    import jieba
    logger.info("✅ Loaded: jieba")
except ImportError:
    logger.error("❌ Thiếu thư viện jieba")
    exit(1)

class ChinesePipeline:
    def __init__(self, args):
        self.args = args
        self.root = os.path.abspath(args.data_root)
        self.raw_dir = os.path.join(self.root, "ALT")

        self.src_path = os.path.join(self.raw_dir, args.src_file)
        self.tgt_path = os.path.join(self.raw_dir, args.tgt_file)

        self.out_dir = os.path.join(self.root, "ALT_Chinese")
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
            logger.error(f"❌ File missing")
            exit(1)

        raw_data = []
        with open(self.src_path, 'r', encoding='utf-8') as fs, open(self.tgt_path, 'r', encoding='utf-8') as ft:
            for s, t in zip(fs, ft):
                s, t = self.normalize(s), self.normalize(t)
                if s and t: raw_data.append((s, t))

        # --- DEDUPLICATION ---
        original_count = len(raw_data)
        data = list(set(raw_data))
        logger.info(f"Original: {original_count} -> Unique: {len(data)}")
        # ---------------------

        random.seed(42)
        random.shuffle(data)
        n = len(data)
        splits = {'train': data[:int(n*0.9)], 'dev': data[int(n*0.9):int(n*0.95)], 'test': data[int(n*0.95):]}

        for mode, items in splits.items():
            with open(os.path.join(self.split_dir, f"{mode}.{self.args.src}"), 'w', encoding='utf-8') as fs, \
                 open(os.path.join(self.split_dir, f"{mode}.{self.args.tgt}"), 'w', encoding='utf-8') as ft:
                for s, t in items:
                    fs.write(s + '\n')
                    ft.write(t + '\n')

    def step_2_tokenize(self):
        logger.info("--- [2] TOKENIZE ---")
        modes = ['train', 'dev', 'test']
        for mode in modes:
            # Tok ZH (jieba)
            inp_s = os.path.join(self.split_dir, f"{mode}.{self.args.src}")
            out_s = os.path.join(self.temp_dir, f"{mode}.tok.{self.args.src}")
            with open(inp_s, 'r', encoding='utf-8') as fi, open(out_s, 'w', encoding='utf-8') as fo:
                for line in tqdm(fi, desc=f"Tok ZH {mode}"):
                    toks = jieba.cut(line.strip(), cut_all=False)
                    fo.write(" ".join(toks) + "\n")
            # Tok VI
            inp_t = os.path.join(self.split_dir, f"{mode}.{self.args.tgt}")
            out_t = os.path.join(self.temp_dir, f"{mode}.tok.{self.args.tgt}")
            with open(inp_t, 'r', encoding='utf-8') as fi, open(out_t, 'w', encoding='utf-8') as fo:
                for line in tqdm(fi, desc=f"Tok VI {mode}"):
                    fo.write(ViTokenizer.tokenize(line.strip()) + "\n")

    def step_3_clean(self):
        logger.info("--- [3] CLEAN ---")
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
                if ls < 1 or lt < 1 or ls > 400 or lt > 400 or ls/lt > 4.0 or lt/ls > 4.0:
                    continue
                os_f.write(s + '\n')
                ot_f.write(t + '\n')
                kept += 1
        logger.info(f"-> Kept: {kept}")

    def step_4_separate_bpe(self):
        logger.info(f"--- [4] SEPARATE BPE (Vocab={self.args.vocab_size}) ---")
        train_src = os.path.join(self.temp_dir, f"train.clean.{self.args.src}")
        train_tgt = os.path.join(self.temp_dir, f"train.clean.{self.args.tgt}")
        
        prefix_src = os.path.join(self.final_dir, f"spm_{self.args.src}")
        prefix_tgt = os.path.join(self.final_dir, f"spm_{self.args.tgt}")

        # ZH: Character coverage 0.9995 để tránh ký tự rác, focus vào Hán tự chuẩn
        logger.info(f"Training BPE {self.args.src}...")
        spm.SentencePieceTrainer.train(
            input=train_src, model_prefix=prefix_src, vocab_size=self.args.vocab_size, 
            model_type='bpe', character_coverage=0.9995 
        )
        
        # VI: 1.0 coverage
        logger.info(f"Training BPE {self.args.tgt}...")
        spm.SentencePieceTrainer.train(
            input=train_tgt, model_prefix=prefix_tgt, vocab_size=self.args.vocab_size, 
            model_type='bpe', character_coverage=1.0
        )

        sp_s = spm.SentencePieceProcessor(model_file=f"{prefix_src}.model")
        sp_t = spm.SentencePieceProcessor(model_file=f"{prefix_tgt}.model")

        for mode in ['train', 'dev', 'test']:
            prefix = "train.clean" if mode == 'train' else f"{mode}.tok"
            with open(os.path.join(self.temp_dir, f"{prefix}.{self.args.src}"), 'r', encoding='utf-8') as fi, \
                 open(os.path.join(self.final_dir, f"{mode}.bpe.{self.args.src}"), 'w', encoding='utf-8') as fo:
                for line in fi: fo.write(" ".join(sp_s.encode_as_pieces(line.strip())) + "\n")
            with open(os.path.join(self.temp_dir, f"{prefix}.{self.args.tgt}"), 'r', encoding='utf-8') as fi, \
                 open(os.path.join(self.final_dir, f"{mode}.bpe.{self.args.tgt}"), 'w', encoding='utf-8') as fo:
                for line in fi: fo.write(" ".join(sp_t.encode_as_pieces(line.strip())) + "\n")
        
        logger.info(f"✅ DONE. Saved to: {self.final_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--src_file", default="ALT.vi-zh.zh")
    parser.add_argument("--tgt_file", default="ALT.vi-zh.vi")
    parser.add_argument("--src", default="zh")
    parser.add_argument("--tgt", default="vi")
    # ĐÃ ĐỔI MẶC ĐỊNH THÀNH 4000
    parser.add_argument("--vocab_size", type=int, default=4000)
    args = parser.parse_args()

    pipeline = ChinesePipeline(args)
    pipeline.step_1_split()
    pipeline.step_2_tokenize()
    pipeline.step_3_clean()
    pipeline.step_4_separate_bpe()