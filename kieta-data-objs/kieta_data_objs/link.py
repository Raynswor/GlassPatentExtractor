from typing import Dict

from . import DocObj


class Link(DocObj):
    """
    Link between two areas, can be directional
    """
    def __init__(self, category: str, frm: str, to: str, directed: bool = False, link_id: str=None) -> None:
        if not link_id:
            self.oid: str = frm + to
        else:
            self.oid: str = link_id
        self.category: str = category
        self.frm: str = frm
        self.to: str = to
        self.directed: bool = directed

    @staticmethod
    def from_dic(d: Dict):
        pass
        # return Page(d['id'], d['number'], d['img'])