from typing import List, Set, Dict, Set

from . import DocObj


class Revision(DocObj):
    def __init__(self, timestamp: str, 
                 objects: Set[str] = None, 
                 comment: str = "", 
                 del_objs: Set[str] = None,
                 reference_revoked: bool = False):
        self.timestamp: str = timestamp
        self.objects: Set[str] = set(objects) if objects else set()
        self.comment: str = comment
        self.del_objs: Set[str] = set(del_objs) if del_objs else set()
        self.reference_revoked: bool = reference_revoked

    def adjust_objs(self, other: Set[str]) -> Set[str]:
        return set(self.objects) + other - set(self.del_objs)
    
    @staticmethod
    def from_dic(d: Dict):
        return Revision(d['timestamp'], d.get('objects',set()) , d.get('comment', ""), 
                        d.get('del_objs', None), d.get('reference_revoked', False))

    def to_dic(self) -> Dict:
        d = {
            'timestamp': self.timestamp,
            'objects': self.objects,
            'comment': self.comment,
            'reference_revoked': self.reference_revoked
        }
        if self.del_objs:
            d['del_objs'] = self.del_objs
        return d