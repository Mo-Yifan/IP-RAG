# src/retrieval/parsers/drugbank_xml.py

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from .base_parser import BaseParser


@dataclass
class Drug:
    drugbank_id: str
    name: str
    description: Optional[str] = None
    indications: Optional[str] = None
    pharmacodynamics: Optional[str] = None
    mechanism_of_action: Optional[str] = None
    toxicity: Optional[str] = None
    metabolism: Optional[str] = None
    half_life: Optional[str] = None
    fda_approved: bool = False

    def has_clinical_text(self) -> bool:
        """判断是否包含任何临床相关文本（用于过滤无用条目）"""
        fields = [
            self.description, self.indications, self.pharmacodynamics,
            self.mechanism_of_action, self.toxicity, self.metabolism, self.half_life
        ]
        return any(field and field.strip() for field in fields)


class DrugBankXMLParser(BaseParser):
    def __init__(self):
        self.ns = {'db': 'http://www.drugbank.ca'}

    def parse(self, xml_path: Path) -> List[Drug]:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        drugs = []

        for drug_elem in root.findall('db:drug', self.ns):
            drug = self._parse_drug(drug_elem)
            if drug and drug.has_clinical_text():  # 只保留有临床信息的药物
                drugs.append(drug)

        return drugs

    def _parse_drug(self, drug_elem: ET.Element) -> Optional[Drug]:
        try:
            dbid_elem = drug_elem.find('db:drugbank-id[@primary="true"]', self.ns)
            if dbid_elem is None or not dbid_elem.text:
                return None
            drugbank_id = dbid_elem.text.strip()

            name_elem = drug_elem.find('db:name', self.ns)
            if name_elem is None or not name_elem.text:
                return None
            name = name_elem.text.strip()

            groups = drug_elem.find('db:groups', self.ns)
            fda_approved = False
            if groups is not None:
                for group in groups.findall('db:group', self.ns):
                    if group.text == "approved":
                        fda_approved = True
                        break

            def get_text(field_name: str) -> Optional[str]:
                elem = drug_elem.find(f'db:{field_name}', self.ns)
                if elem is not None and elem.text:
                    # 清理多余空白和换行
                    return " ".join(elem.text.split())
                return None

            return Drug(
                drugbank_id=drugbank_id,
                name=name,
                description=get_text("description"),
                indications=get_text("indication"),
                pharmacodynamics=get_text("pharmacodynamics"),
                mechanism_of_action=get_text("mechanism-of-action"),
                toxicity=get_text("toxicity"),
                metabolism=get_text("metabolism"),
                half_life=get_text("half-life"),
                fda_approved=fda_approved
            )

        except Exception as e:
            print(f"⚠️  解析药物失败 (ID: {locals().get('drugbank_id', 'unknown')}): {e}")
            return None