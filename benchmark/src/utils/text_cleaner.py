import re
from opencc import OpenCC

try:
    import cn2an
except ImportError:
    cn2an = None

class TextCleaner:
    def __init__(self, config_str='s2t'):
        self.cc = OpenCC(config_str)

    def clean(self, text: str) -> str:
        """強化版文字清理：數字轉中文、轉正體、移除標點符號、轉小寫、去除所有空格"""
        if not text:
            return ""
            
        # 1. 數字正規化：將字串中的阿拉伯數字轉為中文數字 (例如 "18世紀" -> "十八世紀")
        if cn2an is not None:
            try:
                text = cn2an.transform(text, "an2cn")
            except Exception:
                pass # 萬一轉換發生例外，就保持原樣繼續處理
                
        # 2. 轉正體中文
        text = self.cc.convert(text)
        
        # 3. 移除常見標點符號 (新增書名號《》、半形冒號:、半形減號-、括號及其他常見排版符號)
        text = re.sub(r'[，。！？：；「」『』、（）—─""\'’.《》〈〉：\-─–—…\[\]\(\)\{\}#@&*=+<>~|\\`“”‘’]', '', text)
        
        # 4. 轉小寫 (針對英文單字)
        text = text.lower()
        
        # 5. 徹底移除所有空白字元 (\s+ 包含空格、換行、tab 等)
        text = re.sub(r'\s+', '', text)
        return text
