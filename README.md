🌏 Fine-tuning Transformer for Low-Resource Languages (ALT Dataset)

Dự án này tập trung vào việc nghiên cứu và cải thiện chất lượng dịch máy cho các cặp ngôn ngữ ít tài nguyên (Trung-Việt, Khmer-Việt, Lào-Việt) bằng phương pháp Transfer Learning. Dự án được fork và phát triển tiếp nối từ toolkit MultilingualMT-UET-KC4.1.

📖 Giới thiệu chung

Mục tiêu cốt lõi của đề tài là:

Huấn luyện thành công mô hình Transformer sâu trên cấu hình phần cứng hạn chế (GPU 4GB - 8GB VRAM).

Ứng dụng kỹ thuật Fine-tuning từ các mô hình Pre-trained để vượt qua rào cản thiếu hụt dữ liệu của các ngôn ngữ khu vực Đông Nam Á.

🛠 Cài đặt môi trường

Đảm bảo hệ thống của bạn đã cài đặt Python >= 3.8 và hỗ trợ GPU (CUDA).

# 1. Clone repository
git clone [https://github.com/chiendz11/NMT.git](https://github.com/chiendz11/NMT.git)
cd NMT

# 2. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 3. Cài đặt công cụ đánh giá chuẩn SacreBLEU
pip install sacrebleu


📊 Dữ liệu thực nghiệm

Dự án sử dụng tập dữ liệu ALT (Asian Language Treebank) đã qua tiền xử lý tách từ và BPE (Byte Pair Encoding):

Nguồn (Source): Tiếng Trung (.zh), Tiếng Khmer (.km), Tiếng Lào (.lo).

Đích (Target): Tiếng Việt (.vi).

🚀 Hướng dẫn sử dụng

1. Huấn luyện (Fine-tuning)

Sử dụng cấu hình YAML đã được tối ưu hóa cho bộ nhớ GPU thấp (Batch size nhỏ, tích lũy gradient).

python -m bin.main train --model Transformer \
    --model_dir ./models/alt_lo_finetune_transformer \
    --config ./config/alt_finetune_lo_prototype.yml


2. Suy luận (Inference)

Dịch văn bản từ ngôn ngữ nguồn sang tiếng Việt bằng mô hình đã huấn luyện.
Lưu ý: Code đã được cập nhật để tự động tạo thư mục lưu kết quả nếu chưa tồn tại.

python -m bin.main infer --model Transformer \
    --model_dir ./models/alt_lo_finetune_transformer \
    --features_file ./data/ALT_Laos/test.bpe.lo \
    --predictions_file ./data/predictions/predictions_lo2vi_transformer_finetune_alt


3. Đánh giá chất lượng (SacreBLEU)

Sử dụng thư viện chuẩn SacreBLEU để đánh giá độ chính xác của bản dịch so với nhãn gốc (Reference). Không dùng script Perl cũ.

Cú pháp:
sacrebleu [File_Đáp_Án] -i [File_Máy_Dịch] -m bleu -b -w 4

Lệnh mẫu:

sacrebleu ./data/ALT_Laos/test.bpe.vi \
    -i ./data/predictions/predictions_lo2vi_transformer_finetune_alt \
    -m bleu -b -w 4


📈 Kết quả đạt được

Việc áp dụng Fine-tuning giúp cải thiện điểm BLEU vượt trội so với việc huấn luyện từ đầu (Train from Scratch):

Cặp ngôn ngữ

Baseline (Scratch)

Fine-tuning (Pre-train)

Cải thiện (Δ)

Trung → Việt

18.91

22.06

+3.15

Khmer → Việt

24.42

26.46

+2.04

Lào → Việt

18.41

22.07

+3.36

📝 Kết luận & Hướng phát triển

Thành tựu: Chứng minh Transfer Learning cực kỳ hiệu quả cho ngôn ngữ ít tài nguyên. Tối ưu hóa pipeline huấn luyện thành công trên GPU phổ thông.

Hạn chế: Còn gặp lỗi dịch sai với các thuật ngữ chuyên ngành hẹp (hành chính/khoa học) do dữ liệu ALT còn hạn chế về miền từ vựng.

Tương lai:

Triển khai Back-translation để tự động hóa mở rộng dữ liệu (Data Augmentation).

Áp dụng Quantization (INT8) để tăng tốc độ dịch trên CPU.

🤝 Thông tin liên hệ

Sinh viên thực hiện: Bùi Anh Chiến

Giảng viên hướng dẫn: TS. Trần Hồng Việt (thviet@vnu.edu.vn)

GitHub Collaboration: Đã mời thviet79@gmail.com làm cộng tác viên.