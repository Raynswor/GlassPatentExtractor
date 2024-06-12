import base64
import cProfile
import datetime
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import fitz

from kieta_data_objs import Document, Area, Page, BoundingBox, Revision, Font

from kieta_modules import Module
from kieta_modules.util import nms_merge


def clean_text(st):
    return st.replace("𝐵", "b").replace("ﬁ", "fi")



class PDFConverter(Module):
    _MODULE_TYPE = "PDFConverter"

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = None) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.dpi = int(parameters.get("dpi", 300))
        self.keep_chars = parameters.get("keep_chars", False)
        self.extract_drawn_figures = parameters.get(
            "extract_drawn_figures", False)

    def execute(self, input: tempfile.NamedTemporaryFile) -> Document:
        # pr.enable()
        doc: Document = self.convert_xml_with_pymupdf(input["id"], input["file"], self.dpi)
        # pr.disable()

        # s = io.StringIO()
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())

        # for p in doc.pages.byId.values():
        #     doc.references.byId[p.oid] += [y.oid for y in drawn_lines[p.number]]
        #     for x in drawn_lines[p.number]:
        #         doc.areas.append(x)
        #         doc.areas[x.oid].boundingBox.img_sp = True
        #         # doc.areas.allIds.append(x.oid)

        #     # check if majority of page content is vertical --> rotate
        #     no_vertical = 0
        #     for a in doc.references[p.oid]:
        #         if doc.areas[a].category != "Line":
        #             continue
        #         if doc.areas[a].boundingBox.width < doc.areas[a].boundingBox.height:
        #             no_vertical += 1
        #     if no_vertical > len(doc.references[p.oid]) * 0.5:
        #         doc.pages[p.oid].set_rotation(90)
        #         for a in doc.references[p.oid]:
        #             doc.areas[a].boundingBox.transpose()
        # doc.raw_pdf = b.read()
        doc.metadata = {"created": datetime.datetime.now().isoformat()}
        try:
            doc.oid = input["oid"]
        except KeyError:
            doc.oid = input["id"]
        return doc

    def convert_xml_with_pymupdf(self, name, stream, dpi=300) -> Document:
        document: Document = Document(name)
        with fitz.open(stream=stream, filetype="pdf") as doc:
            # Extract XML content
            for p_ix, page in enumerate(doc):
                pixmap = page.get_pixmap(dpi=dpi)

                # ensure, that the image is not too large
                if pixmap.width > 9000:
                    factor = 9000 / pixmap.width
                    pixmap = page.get_pixmap(dpi=round(dpi * factor))
                if pixmap.height > 9000:
                    factor = 9000 / pixmap.height
                    pixmap = page.get_pixmap(dpi=round(dpi * factor))

                factor_width = pixmap.width / page.rect.width
                factor_height = pixmap.height / page.rect.height
                page_obj = Page(
                    f"page-{p_ix}",
                    number=p_ix,
                    img=base64.b64encode(pixmap.tobytes()).decode("utf-8"),
                    img_width=pixmap.width,
                    img_height=pixmap.height,
                    xml_height=page.rect.height,
                    xml_width=page.rect.width,
                    factor_width=factor_width,
                    factor_height=factor_height,
                )
                document.pages.append(page_obj)

                # document.references.allIds.append(f"page-{p_ix}")
                document.references.byId[f"page-{p_ix}"] = list()

                for b in page.get_text("rawdict")["blocks"]:
                    if "lines" not in b:  # image block
                        document.add_area(
                            f"page-{p_ix}", "Image", self.quad_to_boundingBox(b["bbox"]), data={"img": b['image'], 'width': b['width'], 'height': b['height']})
                        continue

                    block_id = document.add_area(
                        f"page-{p_ix}", "Block", self.quad_to_boundingBox(b["bbox"]),convert_to_xml=True)
                    
                    pass
                    # TODO: why do i need this again?
                    # while True:
                    #     merge = False
                    #     for one in range(len(b["lines"])):
                    #         for two in range(one + 1, len(b["lines"])):
                    #             if b["lines"][one]["bbox"][2] > b["lines"][two]["bbox"][0]:
                    #                 b["lines"][one]["bbox"] = [
                    #                     b["lines"][one]["bbox"][0],
                    #                     b["lines"][one]["bbox"][1],
                    #                     b["lines"][two]["bbox"][2],
                    #                     b["lines"][two]["bbox"][2]
                    #                 ]
                    #                 b["lines"][one]["spans"] += b["lines"][two]["spans"]
                    #                 b["lines"].pop(two)
                    #                 merge = True
                    #                 break
                    #         if merge:
                    #             break
                    #     if not merge:
                    #         break

                    for l in b["lines"]:
                        line_id = document.add_area(
                            f"page-{p_ix}", "Line", self.quad_to_boundingBox(l["bbox"]), referenced_by=block_id,convert_to_xml=True
                        )

                        # calculate bounding box new
                        new_line_bb = [10000, 10000, 0, 0]

                        # merge spans that are overlapping
                        prev_size = 0
                        prev_area = None
                        ended_with_umlaut_points = False

                        char_stack = list()
                        for s in l["spans"]:
                            for char in s["chars"]:
                                if char["c"] == " " or char["c"] == "":
                                    char_stack.append(None)
                                    continue
                                
                                if self.keep_chars:
                                    char_stack.append(document.add_area(
                                        f"page-{p_ix}",
                                        "Char",
                                        self.quad_to_boundingBox(char["bbox"]),
                                        data={"content": char["c"]},
                                        convert_to_xml=True
                                    ))
                                else:
                                    char_stack.append(
                                        Area("", "Char", self.quad_to_boundingBox(char["bbox"]), data={"content": char["c"]})
                                    )
                            char_stack.append(None)
                            try:
                                while char_stack[0] is None:
                                    char_stack = char_stack[1:]
                            except IndexError:
                                continue
                            # make for each char group a string
                            while (t := char_stack.index(None)):
                                if self.keep_chars:
                                    strng = [document.get_area_obj(
                                    ar) for ar in char_stack[:t]]
                                else:
                                    strng = char_stack[:t]
                                char_stack = char_stack[t+1:]

                                if len(strng) > 0:
                                    bb = BoundingBox(
                                        strng[0].boundingBox.x1,
                                        strng[0].boundingBox.y1,
                                        strng[-1].boundingBox.x2,
                                        strng[-1].boundingBox.y2,
                                    )
                                    content = "".join(
                                        [c.data["content"] for c in strng])

                                    if s["size"] <= prev_size * 0.75:
                                        # is sub-/ superscript --> add to previous area
                                        prev_area = document.get_area_obj(
                                            prev_area)
                                        if prev_area:
                                            # add symnbol to signal sub or superscript depending on y position
                                            if bb.y2 > prev_area.boundingBox.y2 or bb.y1 > prev_area.boundingBox.y1:
                                                prev_area.data["content"] += "_{"
                                            else:
                                                prev_area.data["content"] += "^{"
                                            prev_area.data["content"] += f"{content.strip()}"
                                            prev_area.data["content"] += "}"
                                            # expand bounding box, but only in x direction
                                            prev_area.boundingBox.x2 = max(
                                                prev_area.boundingBox.x2, bb.x2)
                                            
                                            if self.keep_chars:
                                                document.references.byId[prev_area.oid] += [
                                                    c.oid for c in strng]
                                            prev_area = prev_area.oid
                                    elif ended_with_umlaut_points:  # special exception for umlauts
                                        prev_area = document.get_area_obj(
                                            prev_area)
                                        if prev_area:
                                            prev_area.data["content"] += content.strip()
                                            prev_area.boundingBox.x2 = max(
                                                prev_area.boundingBox.x2, bb.x2)
                                            if self.keep_chars:
                                                document.references.byId[prev_area.oid] += [
                                                    c.oid for c in strng]
                                            prev_area = prev_area.oid
                                            ended_with_umlaut_points = False
                                            content = ""
                                    else:
                                        prev_area = document.add_area(
                                            f"page-{p_ix}",
                                            "String",
                                            bb,
                                            data={
                                                "content": content.strip(),
                                                "font": s["font"],
                                                "size": s["size"],
                                            },
                                            referenced_by=line_id,
                                            references=[c.oid for c in strng] if self.keep_chars else None,
                                            convert_to_xml=True
                                        )
                                    
                                    new_line_bb = [
                                        min(new_line_bb[0], bb.x1),
                                        min(new_line_bb[1], bb.y1),
                                        max(new_line_bb[2], bb.x2),
                                        max(new_line_bb[3], bb.y2),
                                    ]

                                    if content.endswith("¨"):
                                        ended_with_umlaut_points = True
                                if not char_stack:
                                    break
                            if not any(char_stack):
                                char_stack = list()
                            prev_size = s["size"]
                        
                        document.get_area_obj(line_id).boundingBox = BoundingBox(
                            new_line_bb[0], new_line_bb[1], new_line_bb[2], new_line_bb[3], img_sp=False)

                # get images
                xreflist = set()
                for img in doc.get_page_images(p_ix):
                    xref = img[0]
                    if xref in xreflist:
                        continue
                    xreflist.add(xref)
                    try:
                        bbox = page.get_image_rects(img)[0]
                    except IndexError:
                        continue
                    document.add_area(
                        f"page-{p_ix}",
                        "Image",
                        boundingBox=BoundingBox(
                            bbox.x0, bbox.y0, bbox.x1, bbox.y1, img_sp=False), convert_to_xml=True
                    )

                # get drawn lines
                drawn_objects = list()
                if p_ix != 5:
                    continue
                for draw in page.get_drawings():
                    current_drawing = list()

                    add_info = {k: v for k, v in draw.items() if v}

                    try:
                        del add_info["rect"]
                    except KeyError:
                        pass
                    try:
                        del add_info["items"]
                    except KeyError:
                        pass
                    try:
                        del add_info["layer"]
                        del add_info["seqno"]
                    except KeyError:
                        pass

                    for i in draw.get("items", []):
                        data = None
                        item_type = i[0]
                        items = i[1:]

                        to_be_added = list()

                        if item_type == "l":  # Line
                            tp = "DrawnLine"
                            boundingBox = BoundingBox(
                                items[0].x, items[0].y, items[1].x, items[1].y, img_sp=False)
                            data = {"pts": [(items[0].x, items[0].y), (items[1].x, items[1].y)]}

                            to_be_added.append((tp, boundingBox, data))
                        elif item_type == "re":  # Rectangle
                            tp = "DrawnRectangle"
                            boundingBox = BoundingBox(
                                i[1][0], i[1][1], i[1][2], i[1][3], img_sp=False)

                            to_be_added.append((tp, boundingBox, data))
                        elif item_type == "c":  # Bezier
                            min_x, min_y, max_x, max_y = (
                                min(items[0].x, items[1].x, items[2].x, items[3].x),
                                min(items[0].y, items[1].y, items[2].y, items[3].y),
                                max(items[0].x, items[1].x, items[2].x, items[3].x),
                                max(items[0].y, items[1].y, items[2].y, items[3].y),
                            )
                            tp = "DrawnBezier"
                            boundingBox = BoundingBox(
                                min_x, min_y, max_x, max_y, img_sp=False)
                            data = {"pts": [(items[0].x, items[0].y), (items[1].x, items[1].y), (items[2].x, items[2].y), (items[3].x, items[3].y)]}

                            to_be_added.append((tp, boundingBox, data))
                        elif item_type == "qu":  # Quad
                            # make four lines instead
                            # has only one item
                            tp = "DrawnLine"
                            to_be_added.append((tp, BoundingBox(items[0].ll.x, items[0].ll.y,
                                                        items[0].ur.x, items[0].ur.y, img_sp=False), {"pts": [(items[0].ll.x, items[0].ll.y), (items[0].ul.x, items[0].ul.y)]}))
                            to_be_added.append((tp, BoundingBox(items[0].ul.x, items[0].ul.y,
                                                        items[0].lr.x, items[0].lr.y, img_sp=False), {"pts": [(items[0].ul.x, items[0].ul.y), (items[0].ur.x, items[0].ur.y)]}))
                            to_be_added.append((tp, BoundingBox(items[0].ll.x, items[0].ll.y,
                                                        items[0].ul.x, items[0].ul.y, img_sp=False), {"pts": [(items[0].ll.x, items[0].ll.y), (items[0].lr.x, items[0].lr.y)]}))
                            to_be_added.append((tp, BoundingBox(items[0].lr.x, items[0].lr.y,
                                                        items[0].ur.x, items[0].ur.y, img_sp=False), {"pts": [(items[0].lr.x, items[0].lr.y), (items[0].ur.x, items[0].ur.y)]}))

                        for tp, boundingBox, data in to_be_added:
                            current_drawing.append(
                                document.add_area(
                                        f"page-{p_ix}",
                                        tp,
                                        boundingBox,
                                        data=data,
                                        convert_to_xml=True
                                    )
                            )
                        print(current_drawing[-1])
                    # make bb for drawn group
                    bbs = [document.get_area_obj(
                        bb).boundingBox for bb in current_drawing]
                    if not bbs:
                        continue
                    bb = bbs[0] 
                    for b in bbs[1:]:
                        bb.expand(b)
                    # add draw group
                    drawn_objects.append(
                        document.add_area(
                            f"page-{p_ix}", "DrawnStroke", bb, references=current_drawing, data=add_info, convert_to_xml=True)
                    )

                if self.extract_drawn_figures:
                    # inferre if there is a group of drawn objects --> that is a figure
                    bbs = [document.get_area_obj(
                        obj).boundingBox for obj in drawn_objects]
                    # check which of them overlap
                    len_before = len(bbs)
                    break_counter = 0
                    while True:
                        bbs = nms_merge(bbs, 0)
                        if len(bbs) == len_before:
                            if break_counter > 3:
                                break
                            else:
                                break_counter += 1
                        len_before = len(bbs)
                    # debug
                    for bb in bbs:
                        if bb.area() < 10:
                            continue
                        document.add_area(
                            f"page-{p_ix}",
                            "DrawnFigure",
                            bb,
                            references=[obj for obj in drawn_objects if bb.overlap(
                                document.get_area_obj(obj).boundingBox)],
                                convert_to_xml=True
                        )
                
                self.debug_msg(f"Page {p_ix} done - {len(document.areas)} areas")

                # check memory size
                # import sys
                # mem = 0
                # for a in document.areas:
                #     mem += sys.getsizeof(a)
                # print(f"Memory [Area]: {mem / 1024 / 1024} MB")
                # mem = 0
                # for p in document.pages:
                #     mem += sys.getsizeof(p)
                # print(f"Memory [Page]: {mem / 1024 / 1024} MB")

        document.cleanup_references()
        return document

    def quad_to_boundingBox(self, coords: Tuple) -> BoundingBox:
        """
        Convert "upper left, upper right, lower left, lower right" coordinates to bounding box
        """
        return BoundingBox(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]), img_sp=False)

    def exakt_string_boundingBox(self, str_dict: Dict) -> BoundingBox:
        # print(str_dict["bbox"][3],str_dict["bbox"][1], str_dict["size"])
        # assert str_dict["bbox"][3]-str_dict["bbox"][1] > str_dict["size"]
        a = str_dict["ascender"]
        d = str_dict["descender"]
        r = fitz.Rect(str_dict["bbox"])
        o = fitz.Point(str_dict["origin"])  # its y-value is the baseline
        r.y1 = o.y - str_dict["size"] * d / (a - d)
        r.y0 = r.y1 - str_dict["size"]

        return BoundingBox(r.x0, r.y0, r.x1, r.y1, img_sp=False)

    def flag_to_font_characteristic(self, flag: int) -> Dict:
        """
        bit 0: superscripted (20) – not a font property, detected by MuPDF code.
        bit 1: italic (21)
        bit 2: serifed (22)
        bit 3: monospaced (23)
        bit 4: bold (24)
        """
        pass
