from typing import Union, Dict

from . import ObjWithID, PageLayout, TypographicLayout


class Page(ObjWithID):
    def __init__(self, oid: str, number: int, img: Union[str, bytes], xml_width: float = 0, xml_height: float = 0,
                 img_width: float = 0, img_height: float = 0, factor_width: float = 0, factor_height: float = 0,
                 layout: PageLayout = None):
        super().__init__(oid)
        self.number: int = number
        self.img: Union[str, bytes] = img
        self.xml_width: float = xml_width
        self.xml_height: float = xml_height
        self.img_width: float = img_width
        self.img_height: float = img_height
        self.factor_width: float = factor_width
        self.factor_height: float = factor_height
        self.layout: PageLayout = layout if layout is not None else PageLayout()

    def set_rotation(self, rotation: int):
        self.layout.rotation = rotation

    def set_division(self, division: float):
        if division < 0 or division > 1:
            raise ValueError(f"division must be between 0 and 1, not {division}")
        self.layout.division = division

    def set_typographic_layout(self, layout: Union[TypographicLayout, str]):
        if isinstance(layout, str):
            try:
                self.layout.typographic_layout = TypographicLayout[layout]
            except KeyError:
                raise ValueError(f"layout must be of type TypographicLayout, not {layout}")
        else:
            self.layout.typographic_layout = layout

    def is_rotated(self):
        return self.layout.rotation != 0

    def __repr__(self):
        return f"Page({self.oid}, {self.number}, {self.layout})"

    @staticmethod
    def from_dic(d: Dict):
        return Page(d['oid'],
                    d['number'],
                    d['img'],
                    d['xml_width'],
                    d['xml_height'],
                    d['img_width'],
                    d['img_height'],
                    d['factor_width'],
                    d['factor_height'],
                    PageLayout.from_dic(d['layout']))

    def to_dic(self) -> Dict:
        return {
            'oid': self.oid,
            'number': self.number,
            'img': self.img,
            'xml_width': self.xml_width,
            'xml_height': self.xml_height,
            'img_width': self.img_width,
            'img_height': self.img_height,
            'factor_width': self.factor_width,
            'factor_height': self.factor_height,
            'layout': self.layout.to_dic()
        }