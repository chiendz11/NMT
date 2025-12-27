BAO CAO DO AN: FINE-TUNING TRANSFORMER CHO NGON NGU IT TAI NGUYEN (ALT DATASET)
GIOI THIEU CHUNG Du an duoc tiep tuc phat trien boi Bui Anh Chien, dua tren cong cu MultilingualMT-UET-KC4.1 (fork tu phien ban 4.0). Muc tieu chinh la toi uu hoa hieu suat dich may cho cac cap ngon ngu Khmer-Viet va Lao-Viet bang phuong phap Transfer Learning (Fine-tuning).

CAI DAT MOI TRUONG

Yeu cau: Python >= 3.6

Thu vien: pip install -r requirements.txt

Tap lenh cai dat: git clone https://github.com/KCDichDaNgu/KC4.0_MultilingualNMT.git cd KC4.0_MultilingualNMT pip install -r requirements.txt

CHUAN BI DU LIEU Su dung tap du lieu ALT (Asian Language Treebank). Du lieu da duoc xu ly BPE (Byte Pair Encoding).

Du lieu nguon (Source): .lo (Lao), .km (Khmer), .zh (Trung)

Du lieu dich (Target): .vi (Viet)

HUONG DAN CHAY MO HINH

4.1. Huan luyen (Training/Fine-tuning) Su dung ky thuat Fine-tuning tu mo hinh Pre-trained de toi uu cho ngon ngu it tai nguyen tren GPU han che (4GB-8GB VRAM).

Lenh huan luyen mau (Cap Khmer - Viet): python -m bin.main train --model Transformer --model_dir ./models/alt_km_finetune_transformer --config ./config/alt_finetune_km_prototype.yml

4.2. Dich (Inference) Lenh dich mau (Cap Lao - Viet): python -m bin.main infer --model Transformer --model_dir ./models/alt_lo_transformer_fineTune/ --features_file ./data/ALT_Laos/test.bpe.lo --predictions_file data/predictions/predictions_lo2vi_transformer_fineTune_alt

4.3. Danh gia (Evaluation) Su dung multi-bleu.perl de tinh diem BLEU: perl third-party/multi-bleu.perl ./data/ALT_Laos/test.bpe.vi < ./data/predictions/predictions_lo2vi_transformer_alt

KET QUA THUC NGHIEM (DIEM BLEU)

Cap ngon ngu | Baseline (Scratch) | Fine-tuning (Pre-train) | +/-
Zh -> Vi | 18.91 | 22.06 | +3.15 Km -> Vi | 24.42 | 26.46 | +2.04 Lo -> Vi | 18.41 | 22.07 | +3.36
TONG KET & HUONG PHAT TRIEN

Dong gop: Huan luyen thanh cong Transformer sau tren GPU han che. Chung minh duoc hieu qua manh me cua Transfer Learning cho ngon ngu it tai nguyen.

Han che: Du lieu con nhieu va kho khan voi thuat ngu chuyen nganh sau.

Huong tuong lai: Ap dung Back-translation quy mo lon, xay dung mo hinh da ngon ngu (Multilingual) va luong tu hoa (Quantization) de trien khai.

THONG TIN PHU

GitHub Source Code: [\[Link Repository cua ban\]](https://github.com/chiendz11)


Sinh vien thuc hien: Bui Anh Chien

Giang vien huong dan: Tran Hong Viet (thviet@vnu.edu.vn)

GitHub Instructor Invited: thviet79@gmail.com