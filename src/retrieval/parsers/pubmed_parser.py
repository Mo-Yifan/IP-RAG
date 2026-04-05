# src/retrieval/parsers/pubmed_parser.py

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List
from .base_parser import BaseParser


@dataclass
class PubmedArticle:
    pmid: str
    title: str
    abstract: str
    drug_mentions: List[str]  # 可通过 NER 提取，此处简化


class PubmedXMLParser(BaseParser):
    """解析 PubMed XML（用于扩展知识源）"""
    
    def parse(self, xml_path: Path) -> List[PubmedArticle]:
        # 简化实现：仅作占位，实际需结合 NER
        return []