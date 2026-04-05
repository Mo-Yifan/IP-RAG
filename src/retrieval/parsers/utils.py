# src/retrieval/parsers/utils.py

import re
from typing import Optional


def clean_text(text: Optional[str]) -> Optional[str]:
    """清理文本：去多余空白、特殊字符"""
    if not text:
        return None
    # 移除多余空白
    text = " ".join(text.split())
    # 可选：移除参考文献标记 [1], (PMID:123456) 等
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\(PMID:\s*\d+\)', '', text)
    return text.strip() or None


def standardize_field_name(raw_name: str) -> str:
    """标准化字段名（用于多源数据融合）"""
    mapping = {
        "indication": "indications",
        "mechanism": "mechanism_of_action",
        "half life": "half_life"
    }
    return mapping.get(raw_name.lower(), raw_name.lower().replace(" ", "_"))