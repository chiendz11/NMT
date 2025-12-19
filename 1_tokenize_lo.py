import sys
import os
from pyvi import ViTokenizer
from laonlp.tokenize import word_tokenize 

# === CẤU HÌNH ĐƯỜNG DẪN ===
INPUT_DIR = './data/Laos_2023'        # Nơi chứa file gốc
OUTPUT_DIR = './data/Laos_2023' # Nơi chứa file đã tách từ

def tokenize_file(input_path, output_path, lang):
    """
    Hàm đọc file, tách từ theo ngôn ngữ và lưu vào file mới.
    """
    print(f"Dang xu ly ({lang}): {input_path} -> {output_path}")
    
    try:
        # Tạo thư mục output nếu chưa có
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            count = 0
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                
                # Tách từ dựa trên ngôn ngữ
                if lang == 'vi':
                    tokenized_line = ViTokenizer.tokenize(line)
                elif lang == 'lo':
                    words = word_tokenize(line)
                    tokenized_line = " ".join(words)
                else:
                    tokenized_line = line 

                f_out.write(tokenized_line + '\n')
                count += 1
                
        print(f"-> Hoan thanh: {count} dong.")

    except FileNotFoundError:
        print(f"LOI: Khong tim thay file dau vao tai: {input_path}")
    except Exception as e:
        print(f"LOI KHONG XAC DINH: {e}")

if __name__ == "__main__":
    # Danh sách các tập dữ liệu cần xử lý
    datasets = ['train2023', 'dev2023', 'test2023']
    
    for prefix in datasets:
        # Xử lý tiếng Việt
        inp_vi = os.path.join(INPUT_DIR, f"{prefix}.vi")
        out_vi = os.path.join(OUTPUT_DIR, f"{prefix}.token.vi")
        tokenize_file(inp_vi, out_vi, lang='vi')
        
        # Xử lý tiếng Lào
        inp_lo = os.path.join(INPUT_DIR, f"{prefix}.lo")
        out_lo = os.path.join(OUTPUT_DIR, f"{prefix}.token.lo")
        tokenize_file(inp_lo, out_lo, lang='lo')
        
    print("\n=== DA TACH TU XONG TOAN BO DATASET ===")