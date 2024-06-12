from __future__ import annotations

import datetime
import json
from typing import Callable, List, Iterable, Dict, Any, Tuple, Union

import uuid
from PIL import Image
import numpy as np

from . import (
    DocObj,
    NormalizedObj,
    Page,
    Area,
    Link,
    Revision,
    BoundingBox,
    Font,
    Entity,
)
from .util import base64_to_img, img_to_base64

import logging

logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)


class Document(DocObj):
    def __init__(
        self,
        oid: str,
        pages: NormalizedObj[Page] = None,
        areas: NormalizedObj[Area] = None,
        links: NormalizedObj[Link] = None,
        references: NormalizedObj[List[str]] = None,
        revisions: List[Revision] = None,
        fonts: List[Font] = None,
        onto_information: List[Entity] = None,
        metadata: Dict[str, Any] = None,
        raw_pdf: bytes = None,
    ):
        self.oid: str = oid
        self.pages: NormalizedObj[Page] = pages if pages else NormalizedObj()
        self.areas: NormalizedObj[Area] = areas if areas else NormalizedObj()
        self.fonts: List[Font] = fonts if fonts else list()
        self.links: NormalizedObj[Link] = links if links else NormalizedObj()
        self.references: NormalizedObj[List[str]] = (
            references if references else NormalizedObj()
        )
        self.revisions: List[Revision] = (
            revisions
            if revisions
            else [
                Revision(
                    datetime.datetime.now().isoformat(timespec="milliseconds"),
                    set(),
                    comment="Initial Revision",
                )
            ]
        )
        self.onto_information: List[Entity] = (
            onto_information if onto_information else list()
        )
        self.metadata: Dict[str, Any] = metadata if metadata else dict()
        self.raw_pdf: bytes = raw_pdf

    def add_area(
        self,
        page: str,
        category: str,
        boundingBox: BoundingBox,
        area_id: str = None,
        data: Any = None,
        referenced_by: List[str] = None,
        references: List[str] = None,
        confidence: float = None,
        convert_to_xml: bool = False,
        id_prefix: str = "",
    ) -> str:
        if not area_id:
            # check if this ID already exists
            area_id = None

            while not area_id or area_id in self.areas.allIds:
                if id_prefix:
                    area_id = f"{self.oid}-{self.pages.byId[page].number}-{category}-{id_prefix}-{str(uuid.uuid4())[:8]}"
                else:
                    area_id = (
                        f"{self.oid}-{self.pages.byId[page].number}-{category}-{str(uuid.uuid4())[:8]}"
                    )

        if convert_to_xml:
            if boundingBox.img_sp:
                boundingBox = BoundingBox(
                    boundingBox.x1 / self.pages.byId[page].factor_width,
                    boundingBox.y1 / self.pages.byId[page].factor_height,
                    boundingBox.x2 / self.pages.byId[page].factor_width,
                    boundingBox.y2 / self.pages.byId[page].factor_height,
                    True,
                )
        else:
            if not boundingBox.img_sp:
                boundingBox = BoundingBox(
                    boundingBox.x1 * self.pages.byId[page].factor_width,
                    boundingBox.y1 * self.pages.byId[page].factor_height,
                    boundingBox.x2 * self.pages.byId[page].factor_width,
                    boundingBox.y2 * self.pages.byId[page].factor_height,
                    False,
                )

        ar = Area(
            area_id,
            boundingBox=boundingBox,
            category=category,
            data=data if data else {},
            confidence=confidence,
        )
        self.areas.append(ar)
        # self.areas.allIds.append(area_id)
        self.references.byId[page].append(area_id)

        # add to latest revision
        self.revisions[-1].objects.add(area_id)

        if referenced_by:
            if isinstance(referenced_by, str):
                try:
                    self.references.byId[referenced_by].append(area_id)
                except KeyError as e:
                    self.references.byId[referenced_by] = [area_id]
            else:
                for r in referenced_by:
                    try:
                        self.references.byId[r].append(area_id)
                    except KeyError as e:
                        self.references.byId[r] = [area_id]
        if references:
            for r in references:
                try:
                    self.references.byId[area_id].append(r)
                except KeyError as e:
                    self.references.byId[area_id] = [r]
                    # self.references.allIds.append(area_id)

        return area_id


    def cleanup_references(self):
        """
        Remove all references that do not exist in the document
        """
        for page in self.pages.allIds:
            # old = len(self.references.byId[page])
            self.references.byId[page] = [
                x for x in self.references.byId[page] if x in self.areas.allIds
            ]
            # print(f"Removed {old - len(self.references.byId[page])} references from {page}")
        # delete all references that are empty
        # old = len(self.references.byId)
        # for k, v in self.references.byId.items():
        #     print(k, v)
        self.references.byId = {
            k: v for k, v in self.references.byId.items() if v
        }
        # print(f"Removed {old - len(self.references.byId)} empty references")

    def delete_areas(self, area_ids: List[str]):
        for area_id in area_ids:
            self.areas.remove(area_id)
            self.references.remove(area_id)
        
        area_ids_set = set(area_ids)
        for k,v in self.references.byId.items():
            self.references.byId[k] = [x for x in v if x not in area_ids_set]
        
        self.revisions[-1].del_objs.update(area_ids)

    def delete_area(self, area_id: str):
        self.areas.remove(area_id)
        # try:
        #     self.references.allIds.remove(area_id)
        # except ValueError:
        #     pass
        self.references.remove(area_id)

        for ref in self.references.byId.values():
            if area_id in ref:
                ref.remove(area_id)
        # self.revisions[-1].objects.remove(area_id)
        self.revisions[-1].del_objs.add(area_id)

    def replace_area(self, old_id: str, new_area: Area) -> None:
        assert self.areas.byId[old_id].category == new_area.category
        self.areas.byId[old_id] = new_area
        # self.areas.allIds[self.areas.allIds.index(old_id)] = new_area.oid
        for ref in self.references.byId.values():
            if old_id in ref:
                ref[ref.index(old_id)] = new_area.oid
        self.revisions[-1].del_objs.add(old_id)
        # self.revisions[-1].objects[self.revisions[-1]
        #                            .objects.index(old_id)] = new_area.oid

    def replace_area_multiple(self, old_id: str, new_areas: List[Area]) -> None:
        page = self.find_page_of_area(old_id)
        self.replace_area(old_id, new_areas[0])
        for a in new_areas[1:]:
            self.add_area(
                page, a.category, a.boundingBox, data=a.data, confidence=a.confidence
            )

    def find_page_of_area(self, area: Union[str, Area]) -> Union[str, None]:
        if isinstance(area, Area):
            area = area.oid
        for page in self.pages.allIds:
            if area in self.references.byId[page]:
                return page
        return None

    def get_latest_areas(self) -> List[str]:
        l = set()
        for rev in self.revisions:
            l.union(rev.adjust_objs(l))
        return l

    def get_areas_left(self, area_id: Union[str, Area], category: str) -> List[Area]:
        if isinstance(area_id, str):
            area_id = self.areas.byId[area_id]
        page = self.find_page_of_area(area_id)
        if area_id.boundingBox.x1 == 0:
            return []
        else:
            return self.get_areas_at_position(
                page, area_id.boundingBox.x1 - 1, area_id.boundingBox.y1, category
            )

    def get_areas_right(self, area_id: Union[str, Area], category: str) -> List[Area]:
        if isinstance(area_id, str):
            area_id = self.areas.byId[area_id]
        page = self.find_page_of_area(area_id)
        if area_id.boundingBox.x2 == self.pages.byId[page].xml_width:
            return []
        else:
            return self.get_areas_at_position(
                page, area_id.boundingBox.x2 + 1, area_id.boundingBox.y1, category
            )

    def get_areas_above(self, area_id: Union[str, Area], category: str) -> List[Area]:
        if isinstance(area_id, str):
            area_id = self.areas.byId[area_id]
        page = self.find_page_of_area(area_id)
        if area_id.boundingBox.y1 == 0:
            return []
        else:
            return self.get_areas_at_position(
                page, area_id.boundingBox.x1, area_id.boundingBox.y1 - 1, category
            )

    def get_areas_below(self, area_id: Union[str, Area], category: str) -> List[Area]:
        if isinstance(area_id, str):
            area_id = self.areas.byId[area_id]
        page = self.find_page_of_area(area_id)
        if area_id.boundingBox.y2 == self.pages.byId[page].xml_height:
            return []
        else:
            return self.get_areas_at_position(
                page, area_id.boundingBox.x1, area_id.boundingBox.y2 + 1, category
            )

    def get_areas_at_position(
        self, page: str, x: int, y: int, category: str
    ) -> List[Area]:
        ret = list()
        for area_id in self.references.byId[page]:
            area = self.areas.byId[area_id]
            if area.category != category:
                continue
            if (
                area.boundingBox.x1 <= x <= area.boundingBox.x2
                and area.boundingBox.y1 <= y <= area.boundingBox.y2
            ):
                ret.append(area)
        return ret

    def get_area_obj(self, area_id: str) -> Union[Area, None]:
        return self.areas.byId.get(area_id, None)

    def get_area_data_value(self, area: Union[str, Area], val: str = "content") -> List[Any]:
        """
        Get the content of an area. If no content is found, see if there are references to other areas and return their content.
        """
        if not isinstance(area, Area):
            area = self.get_area_obj(area)

        if not area:
            return []
        g = area.data.get(val, None)
        if not g:
            refs = self.references.byId.get(area.oid, None)
            if refs:
                ls = list()
                for ref in refs:
                    g = self.get_area_obj(ref).data.get(val, None)
                    if g:
                        ls.append(g)
                return ls
        else:
            return [g]
        return []

    def add_link(
        self,
        category: str,
        frm: str,
        to: str,
        directed: bool = False,
        link_id: str = None,
        association: str = None,
        page: str = "",
    ):
        if not link_id:
            try:
                link_id = (
                    f"{self.pages.byId[page].number}-{category}-{str(uuid.uuid4())[:3]}"
                )
            except KeyError:
                link_id = f"{page}-{category}-{str(uuid.uuid4())[:3]}"
        li = Link(category, frm, to, directed, link_id)
        self.links.append(li)
        # self.links.allIds.append(link_id)
        self.references.byId[page].append(link_id)
        self.revisions[-1].objects.add(link_id)
        if association:
            try:
                self.references.byId[association].append(link_id)
            except KeyError:
                self.references.byId[association] = [link_id]
        return link_id

    def add_revision(self, name: str):
        self.revisions.append(
            Revision(
                datetime.datetime.now().isoformat(timespec="milliseconds"),
                set(),
                comment=name,
            )
        )

    def add_page(self, page: Page = None, img: str = None):
        if not page:
            if isinstance(img, str):
                open_img = Image.open(img)
            elif isinstance(img, np.ndarray):
                open_img = Image.fromarray(img)
            else:
                raise Exception("Invalid image type")
            page = Page(
                f"page-{str(len(self.pages.allIds))}",
                len(self.pages.allIds),
                img_to_base64(open_img),
                img_width=open_img.width,
                img_height=open_img.height,
                xml_width=open_img.width,
                xml_height=open_img.height,
            )
        self.pages.byId[page.oid] = page
        # self.pages.allIds.append(page.oid)
        self.references.byId[page.oid] = list()
        # self.references.allIds.append(page.oid)
        return page.oid

    def get_area_type(self, category: str, page: str | int = "") -> Iterable[Area]:
        # TODO: CHANGED, IF ERROR OCCURS, CHECK HERE
        return self.get_area_by(lambda x: x.category == category, page)

    def get_area_by(
        self, fun: Callable[[Area], bool], page: str | int = ""
    ) -> Iterable[Area]:
        p = ""
        if page in self.pages.allIds:
            p = page
        elif f"page-{page}" in self.pages.allIds:
            p = f"page-{page}"

        if page != "" and p == "":
            return []
        elif page == "":
            for a in self.areas.byId.values():
                if fun(a):
                    yield a
        else:
            for a in self.references.byId[p]:
                if fun(self.areas.byId[a]):
                    yield self.areas.byId[a]

    def get_img_snippets(
        self, areas: List[Area], padding: Tuple[int, int] = (0, 0), page: str = None
    ) -> List[np.typing.ArrayLike]:
        p = self.find_page_of_area(areas[0]) if not page else page
        img = np.array(base64_to_img(self.pages.byId[p].img))

        for area in areas:
            bb = area.boundingBox.get_in_img_space(
                self.pages.byId[p].factor_width, self.pages.byId[p].factor_height
            )
            yield img[
                int(bb.y1) - padding[1] : int(bb.y2) + padding[1],
                int(bb.x1) - padding[0] : int(bb.x2) + padding[0],
                :,
            ]

    def get_img_snippet(
        self, area_id: str, as_string: bool = True, padding: Tuple[int, int] = (0, 0)
    ) -> Union[str, Image.Image]:
        if not isinstance(area_id, Area):
            area = self.get_area_obj(area_id)
        else:
            area = area_id
        if not area:
            raise Exception(f"Area {area_id} does not exist")
        p = self.find_page_of_area(area)
        return self.get_img_snippet_from_bb(area.boundingBox, p, as_string, padding)

    def get_img_snippet_from_bb(
        self,
        bb: BoundingBox,
        p: str,
        as_string: bool,
        padding: Tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        if bb:
            # check if in image space
            bb = bb.get_in_img_space(
                self.pages.byId[p].factor_width, self.pages.byId[p].factor_height
            )
            if isinstance(self.pages.byId[p].img, str):
                img = base64_to_img(self.pages.byId[p].img)
            else:
                img = self.pages.byId[p].img
            cropped = Image.fromarray(
                np.array(img)[
                    max(int(bb.y1) - padding[1], 0) : min(int(bb.y2) + padding[1], img.size[1]-1),
                    max(int(bb.x1) - padding[0], 0) : min(int(bb.x2) + padding[0], img.size[0]-1),
                    :,
                ]
            )
            return cropped if not as_string else img_to_base64(cropped)
        else:
            raise Exception(f"Invalid bounding box")

    def get_img_page(
        self, page: str, as_string: bool = True
    ) -> Union[str, Image.Image]:
        p = ""
        if page in self.pages.allIds:
            p = page
        if p:
            return (
                self.pages.byId[p].img
                if as_string
                else base64_to_img(self.pages.byId[p].img)
            )
        else:
            return ""

    def transpose_page(self, page_id: str):
        # transpose all objects on page: make vertical objects horizontal and vice versa
        for obj in self.references.byId[page_id]:
            self.areas.byId[obj].boundingBox.transpose()

    @staticmethod
    def from_dic(d: Dict) -> Document:
        if not isinstance(d, dict) or not d.get("oid", None) or not d.get("pages", None) or not d.get("areas", None):
            raise Exception(f"Invalid input: {d}")

        # try:
        if not d.get("fonts", []):
            d["fonts"] = list()
        return Document(
            oid=d["oid"],
            pages=NormalizedObj.from_dic(d["pages"], "pages"),
            areas=NormalizedObj.from_dic(d["areas"], "areas"),
            links=NormalizedObj.from_dic(d["links"], "links"),
            references=NormalizedObj.from_dic(d["references"], "references"),
            revisions=[Revision.from_dic(x) for x in d.get("revisions", set())],
            fonts=[Font.from_dic(x) for x in d.get("fonts", [])],
            # onto_information=[Entity.from_dic(x) for x in d.get('onto_information',[])],
            metadata=d.get("metadata", None),
            raw_pdf=d.get("raw_pdf", None),
        )
        # except Exception as e:
        #     import traceback

        #     print(e, traceback.format_exc())

    def to_dic(self) -> Dict:
        return {
            "oid": self.oid,
            "pages": self.pages.to_dic(),
            "areas": self.areas.to_dic(),
            "links": self.links.to_dic(),
            "references": self.references.to_dic(),
            "revisions": [x.to_dic() for x in self.revisions]
            if self.revisions
            else None,
            "fonts": [x.to_dic() for x in self.fonts] if self.fonts else None,
            "onto_information": [x.to_dic() for x in self.onto_information]
            if self.onto_information
            else None,
            "metadata": self.metadata,
            "raw_pdf": self.raw_pdf,
        }

    def to_json(self) -> str:
        import numpy as np

        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                import base64
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, datetime.datetime):
                    return obj.isoformat()
                if isinstance(obj, set):
                    return list(obj)
                if isinstance(obj, bytes):
                    try:
                        return obj.decode("utf-8")
                    except UnicodeDecodeError:
                        # If the bytes object cannot be decoded into a string using UTF-8,
                        # encode it into base64 instead
                        return base64.b64encode(obj).decode("utf-8")
                return super(NpEncoder, self).default(obj)

        return json.dumps(self.to_dic(), cls=NpEncoder)

    def is_referenced_by(self, area: Union[Area, str], type: List[str] = None) -> bool:
        """
        Check if an area is referenced by any other area
        """
        if isinstance(area, Area):
            area = area.oid
        for ref in self.references.allIds:
            if ref in self.pages.allIds:
                continue
            if type and self.areas.byId[ref].category not in type:
                continue
            if area in self.references.byId[ref]:
                return True
        return False
