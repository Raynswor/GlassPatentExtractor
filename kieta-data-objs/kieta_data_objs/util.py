import base64
import io
from typing import List, Dict

from PIL import Image

from . import DocObj, Area, BoundingBox

class GroupedAreas(DocObj):
    def __init__(self, areas: List[Area]):
        self.areas: List[Area] = areas if areas else []
        self.boundingBox = self.get_boundingBox()

    def __len__(self) -> int:
        return len(self.areas)

    def __iter__(self):
        return iter(self.areas)

    def get_boundingBox(self) -> BoundingBox:
        bb = [10000, 10000, 0, 0]
        for vv in self.areas:
            bb[0] = min(vv.boundingBox.x1, bb[0])
            bb[1]= min(vv.boundingBox.y1, bb[1])
            bb[2]= max(vv.boundingBox.x2, bb[2])
            bb[3] = max(vv.boundingBox.y2, bb[3])
        return BoundingBox(*bb, img_sp=self.areas[0].boundingBox.img_sp)

    @staticmethod
    def from_dic(d: Dict):
        return GroupedAreas(d['id'], d['number'], d['img'])

    def to_dic(self) -> Dict:
        return {
            'areas': [x.to_dic() for x in self.areas]
        }


def base64_to_img(base64_string: str) -> Image:
    try:
        header, encoded = base64_string.split(",", 1)
    except ValueError:
        encoded = base64_string
        
    image_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(image_bytes))


def img_to_base64(image: Image, format: str = "JPEG") -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')