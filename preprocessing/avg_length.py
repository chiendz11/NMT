import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_token_length(file_path, language_name):
    print(f"\n--- ĐANG PHÂN TÍCH: {language_name} ({file_path}) ---")
    
    if not os.path.exists(file_path):
        print(f"❌ Lỗi: Không tìm thấy file {file_path}")
        return

    lengths = []
    over_100 = 0
    over_150 = 0
    over_200 = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Tách line thành các token dựa vào khoảng trắng
            tokens = line.strip().split()
            length = len(tokens)
            lengths.append(length)
            
            if length > 100: over_100 += 1
            if length > 150: over_150 += 1
            if length > 200: over_200 += 1

    lengths = np.array(lengths)
    total_sentences = len(lengths)
    
    print(f"✅ Tổng số câu: {total_sentences}")
    print(f"📊 Thống kê độ dài (Tokens):")
    print(f"   - Trung bình (Mean):   {np.mean(lengths):.2f}")
    print(f"   - Trung vị (Median):   {np.median(lengths):.2f}")
    print(f"   - Ngắn nhất (Min):     {np.min(lengths)}")
    print(f"   - Dài nhất (Max):      {np.max(lengths)}")
    print(f"   - 95th Percentile:     {np.percentile(lengths, 95):.2f} (95% số câu ngắn hơn mức này)")
    print(f"   - 99th Percentile:     {np.percentile(lengths, 99):.2f}")
    
    print(f"\n⚠️ CẢNH BÁO CẮT DỮ LIỆU (TRUNCATION):")
    print(f"   - Số câu > 100 tokens: {over_100} ({over_100/total_sentences*100:.2f}%)")
    print(f"   - Số câu > 150 tokens: {over_150} ({over_150/total_sentences*100:.2f}%)")
    print(f"   - Số câu > 200 tokens: {over_200} ({over_200/total_sentences*100:.2f}%)")

    return lengths

# ==============================================================================
# ĐIỀN ĐƯỜNG DẪN FILE CỦA BẠN VÀO ĐÂY
# ==============================================================================
src_file = 'data/ALT_Lao/02_final_separate/train.bpe.lo' # File BPE tiếng Khmer
trg_file = 'data/ALT_Lao/02_final_separate/train.bpe.vi'  # File BPE tiếng Việt

print("BẮT ĐẦU PHÂN TÍCH...")
try:
    src_lens = analyze_token_length(src_file, "tiếng Lào (SRC)")
    trg_lens = analyze_token_length(trg_file, "TIẾNG VIỆT (TRG)")
    
    # Vẽ biểu đồ đơn giản nếu chạy trên máy local có màn hình
    # plt.hist(src_lens, bins=50, alpha=0.5, label='Lào')
    # plt.hist(trg_lens, bins=50, alpha=0.5, label='Việt')
    # plt.legend(loc='upper right')
    # plt.title('Phân bố độ dài câu (Token Count)')
    # plt.show()
    
except Exception as e:
    print(f"Có lỗi xảy ra: {e}")
    print("Bạn hãy cài numpy: pip install numpy")