import sys
import re
import argparse
from tqdm import tqdm

# Ký tự đặc biệt của SentencePiece (U+2581)
# Code của bạn dùng spm mặc định nên chắc chắn sẽ có ký tự này trong output
SP_SPACE = u'\u2581' 

class PostProcessor:
    def __init__(self, lang):
        self.lang = lang

    def decode_sentencepiece(self, text):
        """
        Bước 1: Nối các mảnh BPE lại chuẩn xác theo logic của SentencePiece.
        Input: " T ôi  y êu  Việt  Nam" (Có ký tự SP_SPACE)
        Output: "Tôi yêu Việt Nam"
        """
        text = text.strip()
        
        # Nếu output model chứa ký tự đặc biệt của SentencePiece
        if SP_SPACE in text:
            # Logic: Thay thế SP_SPACE bằng khoảng trắng, xóa khoảng trắng thừa
            text = text.replace(" ", "")      # Xóa khoảng trắng ngăn cách token
            text = text.replace(SP_SPACE, " ") # Thay ký tự SP bằng dấu cách thường
        else:
            # Fallback: Nếu model output không có ký tự SP (hiếm, nhưng phòng hờ)
            # Giả định ghép đôi @@ (nếu dùng BPE cổ điển) hoặc ghép thẳng
            text = text.replace("@@ ", "")
            # Nếu không có dấu hiệu gì, tạm thời nối liền hoặc giữ nguyên tùy case
            # Với pipeline của bạn, 99% sẽ rơi vào case 'if' ở trên.
            pass
            
        return text.strip()

    def post_process_vietnamese(self, text):
        """
        Xử lý riêng cho Tiếng Việt (Output từ PyVi)
        Input: "Tôi đang học tại Đại_học Bách_Khoa ."
        Output: "Tôi đang học tại Đại học Bách Khoa."
        """
        # 1. Quan trọng nhất: Xóa dấu gạch dưới do PyVi sinh ra
        text = text.replace("_", " ")
        
        return text

    def post_process_scriptio_continua(self, text):
        """
        Xử lý cho Tiếng Trung, Lào, Khmer (Các ngôn ngữ viết liền)
        Vấn đề: Tokenizer (Jieba, LaoNLP) đã chèn dấu cách vào giữa các từ.
        Nhiệm vụ: Xóa dấu cách để văn bản liền mạch trở lại.
        """
        # Logic: Chỉ xóa khoảng trắng nếu 2 bên là ký tự của ngôn ngữ đó.
        # Giữ lại khoảng trắng nếu là Tiếng Anh hoặc Số nằm giữa.
        
        if self.lang == 'zh': # Tiếng Trung
            # Tìm: (Chữ Hán) space (Chữ Hán) -> Xóa space
            pat = re.compile(r'(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])')
            text = pat.sub('', text)
            
        elif self.lang in ['lo', 'km']: # Lào / Khmer
            # Với Lào/Khmer, việc xóa toàn bộ space khá rủi ro vì space đôi khi là ngắt câu.
            # Tuy nhiên, output của machine translation thường tokenize quá đà.
            # Best practice an toàn: Xóa space trước các dấu câu đặc biệt
            text = re.sub(r'\s+([។៕])', r'\1', text)
            
            # (Tùy chọn) Nếu bạn muốn output liền tù tì như văn bản gốc:
            # text = text.replace(" ", "") 
            # Nhưng tôi khuyên nên giữ nguyên logic BPE ghép lại, vì model đã học cách đặt space.
            pass

        return text

    def fix_punctuation_and_capitalize(self, text):
        """Bước làm đẹp cuối cùng (Cosmetics)"""
        # 1. Xóa khoảng trắng trước dấu câu (vd: "Hà Nội ." -> "Hà Nội.")
        text = re.sub(r'\s+([.,;:?!])', r'\1', text)
        
        # 2. Thêm khoảng trắng sau dấu câu nếu bị dính (vd: "Hà Nội.Tôi" -> "Hà Nội. Tôi")
        # (Trừ trường hợp số thập phân 3.5)
        text = re.sub(r'([.,;:?!])(?=[^\s\d])', r'\1 ', text)

        # 3. Viết hoa chữ cái đầu câu
        if text:
            text = text[0].upper() + text[1:]
            
        return text

    def run(self, text):
        # BƯỚC 1: Ghép mảnh (De-BPE)
        text = self.decode_sentencepiece(text)
        
        # BƯỚC 2: Xử lý đặc thù ngôn ngữ
        if self.lang == 'vi':
            text = self.post_process_vietnamese(text)
        elif self.lang in ['zh', 'lo', 'km']:
            text = self.post_process_scriptio_continua(text)
            
        # BƯỚC 3: Trang điểm (Dấu câu + Viết hoa)
        text = self.fix_punctuation_and_capitalize(text)
        
        return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="File kết quả dịch thô (output của model)")
    parser.add_argument("--output", required=True, help="File kết quả đẹp sau khi xử lý")
    parser.add_argument("--lang", required=True, choices=['vi', 'zh', 'lo', 'km', 'en'], help="Ngôn ngữ đích")
    args = parser.parse_args()

    print(f"🚀 Bắt đầu Post-processing cho ngôn ngữ: {args.lang.upper()}")
    
    processor = PostProcessor(lang=args.lang)
    count = 0
    
    with open(args.input, 'r', encoding='utf-8') as f_in, \
         open(args.output, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in):
            if not line.strip():
                f_out.write("\n")
                continue
                
            processed_line = processor.run(line)
            f_out.write(processed_line + "\n")
            count += 1

    print(f"✅ Đã xử lý xong {count} câu.")
    print(f"📄 Kết quả lưu tại: {args.output}")