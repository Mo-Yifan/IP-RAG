# src/retrieval/parsers/drugbank_json.py

import json
from pathlib import Path
from typing import List, Optional
from .base_parser import BaseParser
from .drugbank_xml import Drug  # 复用同一 Drug 类


class DrugBankJSONParser(BaseParser):
    """解析 DrugBank JSON 格式（如果未来使用）"""

    def parse(self, json_path: Path) -> List[Drug]:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        drugs = []
        for item in data.get("drugs", []):
            drug = self._parse_drug(item)
            if drug and drug.has_clinical_text():
                drugs.append(drug)
        return drugs

    def _parse_drug(self, item: dict) -> Optional[Drug]:
        try:
            drugbank_id = item.get("drugbank_id")
            name = item.get("name")
            if not drugbank_id or not name:
                return None

            def safe_get(key: str) -> Optional[str]:
                val = item.get(key)
                return " ".join(str(val).split()) if val else None

            return Drug(
                drugbank_id=drugbank_id,
                name=name,
                description=safe_get("description"),
                indications=safe_get("indications"),
                pharmacodynamics=safe_get("pharmacodynamics"),
                mechanism_of_action=safe_get("mechanism_of_action"),
                toxicity=safe_get("toxicity"),
                metabolism=safe_get("metabolism"),
                half_life=safe_get("half_life"),
                fda_approved=item.get("groups") == ["approved"]
            )
        except Exception as e:
            print(f"⚠️  JSON 解析失败 (ID: {item.get('drugbank_id', 'unknown')}): {e}")
            return None