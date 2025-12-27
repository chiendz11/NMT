🌏 Fine-tuning Transformer for Low-Resource Languages (ALT Dataset)Dự án này tập trung vào việc nghiên cứu và cải thiện chất lượng dịch máy cho các cặp ngôn ngữ ít tài nguyên (Trung-Việt, Khmer-Việt, Lào-Việt) bằng phương pháp Transfer Learning. Dự án được fork và phát triển tiếp nối từ toolkit MultilingualMT-UET-KC4.1.📖 Giới thiệu chungMục tiêu cốt lõi của đề tài là:Huấn luyện thành công mô hình Transformer sâu trên cấu hình phần cứng hạn chế (GPU 4GB-8GB VRAM).Ứng dụng kỹ thuật Fine-tuning từ các mô hình Pre-trained để vượt qua rào cản thiếu hụt dữ liệu của các ngôn ngữ khu vực Đông Nam Á.🛠 Cài đặt môi trườngĐảm bảo hệ thống của bạn đã cài đặt Python >= 3.6.Bash# Clone repository
git clone https://github.com/chiendz11/NMT.git
cd NMT

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
📊 Dữ liệu thực nghiệmDự án sử dụng tập dữ liệu ALT (Asian Language Treebank) đã qua tiền xử lý BPE:Nguồn (Source): Tiếng Trung (.zh), Tiếng Khmer (.km), Tiếng Lào (.lo).Đích (Target): Tiếng Việt (.vi).🚀 Hướng dẫn sử dụng1. Huấn luyện (Fine-tuning)Sử dụng cấu hình YAML để tối ưu cho bộ nhớ GPU thấp.Bashpython -m bin.main train \
    --model Transformer \
    --model_dir ./models/alt_km_finetune_transformer \
    --config ./config/alt_finetune_km_prototype.yml
2. Suy luận (Inference)Dịch văn bản từ ngôn ngữ nguồn sang tiếng Việt:Bashpython -m bin.main infer \
    --model Transformer \
    --model_dir ./models/alt_lo_transformer_fineTune/ \
    --features_file ./data/ALT_Laos/test.bpe.lo \
    --predictions_file data/predictions/predictions_lo2vi_transformer_fineTune_alt
3. Đánh giá chất lượng (BLEU)Bashperl third-party/multi-bleu.perl ./data/ALT_Laos/test.bpe.vi < ./data/predictions/predictions_lo2vi_transformer_alt
📈 Kết quả đạt đượcViệc áp dụng Fine-tuning giúp cải thiện điểm BLEU vượt trội so với huấn luyện từ đầu (Scratch):Cặp ngôn ngữBaseline (Scratch)Fine-tuning (Pre-train)Cải thiện (Δ)Trung → Việt18.9122.06+3.15Khmer → Việt24.4226.46+2.04Lào → Việt18.4122.07+3.36📝 Kết luận & Hướng phát triểnThành tựu: Chứng minh Transfer Learning cực kỳ hiệu quả cho ngôn ngữ ít tài nguyên. Tối ưu hóa pipeline huấn luyện trên GPU phổ thông.Hạn chế: Còn gặp lỗi với các thuật ngữ chuyên ngành hành chính/khoa học do dữ liệu huấn luyện còn nhiễu.Tương lai: Triển khai Back-translation để tự động hóa mở rộng dữ liệu và áp dụng Quantization để tăng tốc độ dịch trên CPU.🤝 Thông tin liên hệSinh viên: Bùi Anh ChiếnGiảng viên hướng dẫn: TS. Trần Hồng Việt (thviet@vnu.edu.vn)GitHub Collaboration: Đã mời thviet79@gmail.com làm cộng tác viên.