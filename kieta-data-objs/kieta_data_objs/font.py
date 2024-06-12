from typing import Any, Dict

from kieta_data_objs import ObjWithID

class Font(ObjWithID):
    def __init__(self, oid: str, color: str, size: float, style: Any = None):
        super().__init__(oid)
        self.color = color
        self.size = size
        self.style = style

    def is_super_script(self) -> bool:
        return "superscript" in self.style

    def is_sub_script(self) -> bool:
        return "subscript" in self.style

    @staticmethod
    def from_dic(d: Dict):
        return Font(d['oid'],
                    d['color'],
                    float(d['size']),
                    d['style']
                    )

    def to_dic(self) -> Dict:
        return {
            'oid': self.oid,
            'color': self.color,
            'size': self.size,
            'style': self.style
        }