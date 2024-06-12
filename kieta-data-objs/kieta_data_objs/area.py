from typing import Any, Optional, Dict

from . import ObjWithID, BoundingBox


class Area(ObjWithID):
    def __init__(self, oid: str, category: str, boundingBox: BoundingBox, data: Any = None, confidence: float = None):
        super().__init__(oid)
        self.category: str = category
        self.boundingBox: BoundingBox = boundingBox
        self.data: Any = data if data else dict()
        self.confidence: Optional[float] = confidence

    def __hash__(self) -> int:
        return hash(self.oid)

    def __repr__(self) -> str:
        return f"<{self.oid}: {self.boundingBox} {self.data if self.data else ''}>"

    def merge(self, other: 'Area', merge_data: bool = True, merge_confidence: bool = True):
        self.boundingBox.expand(other.boundingBox)
        if merge_data:
            # add all new keys
            for k, v in other.data.items():
                if k not in self.data:
                    self.data[k] = v
            # update existing keys
            for k, v in self.data.items():
                if k in other.data:
                    if isinstance(v, list):
                        v.extend(other.data[k])
                    elif isinstance(v, dict):
                        v.update(other.data[k])
                    elif isinstance(v, str):
                        self.data[k] = v + other.data[k]
                    else:
                        self.data[k] = other.data[k]
        if merge_confidence:
            self.confidence = max(self.confidence, other.confidence)

    @staticmethod
    def from_dic(d: Dict):
        return Area(d['oid'],
                    d['category'],
                    BoundingBox.from_dic(d['boundingBox']),
                    data=d.get('data', None),
                    confidence=d.get('confidence', None)
        )

    def to_dic(self) -> Dict:
        dic = {
            'oid': self.oid,
            'category': self.category,
            'boundingBox': self.boundingBox.to_dic(),
        }
        if self.data:
            dic['data'] = self.data
        if self.confidence:
            dic['confidence'] = self.confidence
        return dic
