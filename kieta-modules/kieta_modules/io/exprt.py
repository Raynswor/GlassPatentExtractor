from typing import Optional, Dict
import xml.etree.ElementTree as ET
from abc import abstractmethod
import itertools
import re

from kieta_modules import Module, util

from typing import Any, Dict, Optional, List

from kieta_data_objs import Document


import datetime
from PIL import Image, ImageDraw, ImageFont

import os

import json

# TODO
# - implement write_to_file switch


class ExportModule(Module):
    _MODULE_TYPE = "ExportModule"

    def __init__(self, stage: int, parameters: Dict | None = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.output_dir = parameters.get("output_dir", "output")
        self.format = parameters.get("format", "png")
        self.write_to_file = parameters.get("write_to_file", True)
        self.prefix = parameters.get("prefix", "")

        if self.write_to_file and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def execute(self, inpt: Document) -> Document:
        raise NotImplementedError(
            "ExportModule is abstract and cannot be executed")


class ExportPageXMLModule(ExportModule):
    _MODULE_TYPE = "ExportPageXMLModule"

    def __init__(self, stage: int, parameters: Optional[Dict] = {}, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.confidence_threshold = parameters.get("confidence_threshold", -1)
        self.mode = parameters.get("mode", "text")
        self.text_category = parameters.get("text_category", "String")
        self.line_category = parameters.get("line_category", "Line")
        self.drawn_line_category = parameters.get(
            "drawn_line_category", "DrawnLine")
        self.optimize_baseline = parameters.get("optimize_baseline", False)
        self.bb_pad = parameters.get("bb_pad", 0)

    def execute(self, inpt: Document) -> Document:
        match self.mode:
            case "text":
                strs = self.export_text_mode(inpt)
            case "drawn_line":
                strs = self.export_drawn_line_mode(inpt)
            case _:
                raise ValueError("Invalid mode")

        # write each string to a file and save corresponding image
        for i, s in enumerate(strs):
            with open(f"{self.output_dir}/{self.prefix}{inpt.oid}_page{i}.xml", "w") as f:
                f.write(s)
            inpt.get_img_page(list(inpt.pages.allIds)[i], as_string=False).save(
                f"{self.output_dir}/{self.prefix}{inpt.oid}_page{i}.png")

        return inpt

    def export_text_mode(self, doc: Document) -> List[str]:
        from libpagexml3 import libpagexml3 as pagexml
        pagexml_strings = list()
        for page_id in doc.pages.allIds:
            page = pagexml.PageXMLPage(
                image_filename=page_id,
                created_on=datetime.datetime.now().isoformat(timespec='milliseconds'),
                last_edited=datetime.datetime.now().isoformat(timespec='milliseconds'),
                image_width=doc.pages.byId[page_id].img_width,
                image_height=doc.pages.byId[page_id].img_height,
                regions=list()
            )
            IMG = doc.get_img_page(page_id, as_string=False)
            for area in doc.get_area_type(self.line_category, page=page_id):
                # get text from all areas referenced by this line
                likely_correct = True
                text = []
                for ref in doc.references.byId[area.oid]:
                    ref = doc.get_area_obj(ref)
                    if ref.category == self.text_category:
                        text.append((
                            ref.boundingBox.x1,
                            ref.data.get('content', '')
                        ))

                    # if not ref.confidence or ref.confidence < self.confidence_threshold:
                    #     likely_correct = False
                    # if ref.confidence and ref.confidence > self.confidence_threshold:
                    #     likely_correct = False

                if not likely_correct:
                    continue

                text = sorted(text, key=lambda x: x[0])
                text = " ".join([x[1] for x in text])

                # optimize as follows
                # start from y-middle
                # snip picture 5 pixels after the last connected black pixel in direction of top and bottom
                # Find the last connected black pixel in the direction of top and bottom
                img_space_bb = area.boundingBox.get_in_img_space(
                    doc.pages.byId[page_id].factor_width,
                    doc.pages.byId[page_id].factor_height)
                if self.bb_pad:
                    img_space_bb.x1 -= self.bb_pad
                    img_space_bb.x2 += self.bb_pad
                    img_space_bb.y1 -= self.bb_pad
                    img_space_bb.y2 += self.bb_pad

                if self.optimize_baseline:
                    y_middle = int(img_space_bb.height // 2)
                    last_black_pixel_top = None
                    last_black_pixel_bottom = None

                    for y in range(y_middle, -1, -1):
                        for x in range(int(img_space_bb.width)):
                            # Assuming black pixels are (0, 0, 0)
                            if IMG.getpixel((x, y)) == (0, 0, 0):
                                last_black_pixel_top = y
                                break
                        if last_black_pixel_top is not None:
                            break

                    for y in range(y_middle, int(img_space_bb.height)):
                        for x in range(int(img_space_bb.width)):
                            # Assuming black pixels are (0, 0, 0)
                            if IMG.getpixel((x, y)) == (0, 0, 0):
                                last_black_pixel_bottom = y
                                break
                        if last_black_pixel_bottom is not None:
                            break

                    # Snip the image 5 pixels after the last connected black pixel
                    img_space_bb.y1 = last_black_pixel_bottom - \
                        5 if last_black_pixel_bottom else img_space_bb.y1
                    img_space_bb.y2 = last_black_pixel_top + \
                        5 if last_black_pixel_top else img_space_bb.y2

                baseline = area.data.get('baseline', None)
                if baseline:
                    baseline = [baseline[0], baseline[-1]]

                page.regions.append(pagexml.PageRegion(
                    region_id=area.oid,
                    region_type=pagexml.PageRegionTypes.TEXT_REGION,
                    region_text_type=pagexml.PageXMLTextTypes.PARAGRAPH,
                    polygon=img_space_bb.polygon(),
                    text_lines=[
                        pagexml.TextLine(
                                    id=area.oid,
                                    parent_region_id=area.oid,
                                    polygon=img_space_bb.polygon(),
                                    text_equivs={
                                        pagexml.TextEquiv.TEXT_ID_GROUND_TRUTH: pagexml.TextEquiv(
                                            index=pagexml.TextEquiv.TEXT_ID_GROUND_TRUTH,
                                            text=text
                                        )
                                    },
                                    baseline=baseline
                                    )
                    ]
                ))
            pstr = pagexml.PageXMLBuilder.build_pagexml(
                page=page, creator="SeKe")
            pagexml_strings.append(pstr)
        return pagexml_strings

    def export_drawn_line_mode(self, doc: Document) -> List[str]:
        from libpagexml3 import libpagexml3 as pagexml
        pagexml_strings = list()
        for page_id in doc.pages.allIds:
            page = pagexml.PageXMLPage(
                image_filename=page_id,
                created_on=datetime.datetime.now().isoformat(timespec='milliseconds'),
                last_edited=datetime.datetime.now().isoformat(timespec='milliseconds'),
                image_width=doc.pages.byId[page_id].img_width,
                image_height=doc.pages.byId[page_id].img_height,
                regions=list()
            )
            page_obj = doc.pages.byId[page_id]
            for area in doc.get_area_type(self.drawn_line_category, page=page_id):
                img_space_bb = area.boundingBox.get_in_img_space(
                    page_obj.factor_width, page_obj.factor_height)
                page.regions.append(pagexml.PageRegion(
                    region_id=area.oid,
                    region_type=pagexml.PageRegionTypes.TEXT_REGION,
                    region_text_type=pagexml.PageXMLTextTypes.PARAGRAPH,
                    polygon=img_space_bb.polygon(),
                    text_lines=[
                        pagexml.TextLine(
                                    id=area.oid,
                                    parent_region_id=area.oid,
                                    polygon=img_space_bb.polygon(),
                                    baseline=[
                                        [img_space_bb.x1, img_space_bb.y1],
                                        [img_space_bb.x2, img_space_bb.y2]
                                    ])
                    ]
                ))
            pstr = pagexml.PageXMLBuilder.build_pagexml(
                page=page, creator="SeKe")
            pagexml_strings.append(pstr)
        return pagexml_strings


class ExportJSONModule(ExportModule):
    _MODULE_TYPE = "ExportJSONModule"

    def __init__(self, stage: int, parameters: Optional[Dict] = {}, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.include_pictures = parameters.get("include_pictures", False)

    def execute(self, inpt: Document) -> Document:
        json_str = inpt.to_json()

        if not self.include_pictures:
            # remove all image data "img": "****"
            occs = re.findall(r'"img": "\S*"', json_str)
            for occ in occs:
                json_str = json_str.replace(occ, '"img": ""')

        with open(f"{self.output_dir}/{self.prefix}{inpt.oid}.json", "w") as f:
            f.write(json_str)
        return inpt


class ExportTableContentModule(ExportModule):
    _MODULE_TYPE = "ExportTableContentModule"

    def __init__(self, stage: int, parameters: Optional[Dict] = {}, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.table_category_name = self.parameters.get(
            "table_category_name", "Table")

    def execute(self, inpt: Document) -> Document:
        for p_id in inpt.pages.allIds:
            for area in inpt.get_area_type(self.table_category_name, page=p_id):
                tab_content = []
                try:
                    for ir, row in enumerate(area.data['cells']):
                        curr = list()
                        for ic, cell in enumerate(row):
                            st = ""
                            if cell:
                                if "column_label" in inpt.areas.byId[cell].data:
                                    st += "~"
                                if "row_label" in inpt.areas.byId[cell].data:
                                    st += "^"
                                st += inpt.areas.byId[cell].data['content']
                            curr.append(st)
                        tab_content.append(curr)
                except KeyError:
                    print("No table data found")
                    continue

                with open(f"{self.output_dir}/{self.prefix}{inpt.oid}_page{p_id}_table{area.oid}.{self.format}", "w") as f:
                    match self.format:
                        case "csv":
                            f.write(
                                "\n".join([",".join([f'"{x}"' for x in row]) for row in tab_content]))
                        case "tsv":
                            f.write(
                                "\n".join(["\t".join([f'"{x}"' for x in row]) for row in tab_content]))
                        case "html":
                            # add css to make it look like a table
                            f.write("<style>table {border-collapse: collapse;} table, th, td {border: 1px solid black;}</style>")
                            f.write("<table>")
                            for row in tab_content:
                                f.write("<tr>")
                                for cell in row:
                                    f.write(f"<td>{cell}</td>")
                                f.write("</tr>")
                            f.write("</table>")
                        case "txt":
                            f.write(
                                "\n".join(["\t".join(row) for row in tab_content]))
                        case "md":
                            f.write(
                                "\n".join(["|".join(row) for row in tab_content]))
                        case _:
                            raise ValueError("Invalid table format")
        return inpt


class ExportTableStructureModule(ExportModule):
    _MODULE_TYPE = "ExportTableStructureModule"

    """
    {
        "chunks": [
        {
            "id": "chunk_0",
            "pos": [
            147.96600341796875,
            205.49998474121094,
            475.7929992675781,
            480.4206237792969
            ],
            "text": "Probability"
        }],
        "relations": [{
            from: "chunk_0",
            to: "chunk_1",
            type: "column",
            skips: 0
            }
        ],
        "cells": [
        {
            "id": 21,
            "tex": "one two",
            "content": [
                "chunk_0",
                "chunk_1"
            ],
            "start_row": 5,
            "end_row": 5,
            "start_col": 1,
            "end_col": 1
        },
        "triples": [
            {
            "data": [],
            "column_labels": [],
            "row_labels": []
            }
        ]
        ]

    """

    def __init__(self, stage: int, parameters: Optional[Dict] = {}, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.table_category_name = self.parameters.get(
            "table_category_name", "Table")
        self.elementwise_export = self.parameters.get(
            "elementwise_export", False)
        self.format = "json"

    def execute(self, inpt: Document) -> Document:
        tables = list()

        for p_ix, p_id in enumerate(inpt.pages.allIds):
            for area in inpt.get_area_type(self.table_category_name, page=p_id):
                content = {
                    "meta": {
                        "document": inpt.oid,
                        "page": p_ix+1,
                        "caption": inpt.get_area_obj(area.data.get('caption', None)).data['content'] if area.data.get('caption', None) else "not found",
                    },
                    "chunks": [], "relations": [],
                    "cells": [], "triples": [], "rows": []}
                try:
                    already_seen_cell, already_seen_chunks = set(), set()
                    for ir, row in enumerate(area.data['cells']):
                        current_row = list()
                        for ic, cell in enumerate(row):
                            if not cell:
                                current_row.append({
                                    "content": "",
                                    "column_label": False,
                                    "row_label": False,
                                })
                                continue
                            content["cells"].append({
                                "id": cell,
                                "text": inpt.get_area_obj(cell).data['content'],
                                "content": [inpt.get_area_obj(ccc).data['content'] for ccc in inpt.references.byId[cell]],
                                "start_row": ir,
                                "end_row": ir,
                                "start_col": ic,
                                "end_col": ic,
                                "row_nums": [ir],
                                "column_nums": [ic],
                                "bbox": inpt.get_area_obj(cell).boundingBox.tuple()
                            })

                            cell_obj = inpt.get_area_obj(cell)

                            current_row.append({
                                "content": " ".join([inpt.get_area_obj(ccc).data['content'] for ccc in inpt.references.byId[cell]]),
                                "column_label": cell_obj.data.get('column_label', False),
                                "row_label": cell_obj.data.get('row_label', False),
                            })
                            # triples
                            if not (cell_obj.data.get('column_label') or cell_obj.data.get('row_label')):
                                column_header = [area.data['cells'][ir-counter][ic] for counter in range(1, ir+1) if (
                                    t := inpt.get_area_obj(area.data['cells'][ir-counter][ic])) and t.data.get('column_label')]
                                row_header = [area.data['cells'][ir][ic-counter] for counter in range(1, ic+1) if (
                                    t := inpt.get_area_obj(area.data['cells'][ir][ic-counter])) and t.data.get('row_label')]
                                content["triples"].append({
                                    "data": [inpt.get_area_obj(ccc).data['content'] for ccc in inpt.references.byId[cell]],
                                    "column_labels": [inpt.get_area_obj(ccc).data['content'] for ccc in reversed(column_header)],
                                    "row_labels": [inpt.get_area_obj(ccc).data['content'] for ccc in reversed(row_header)]
                                })

                            if cell in already_seen_cell:
                                # adjust end_row and end_col
                                obj = None
                                for c in content["cells"]:
                                    if c["id"] == cell:
                                        obj = c
                                        break
                                obj["end_row"] = ir
                                obj["end_col"] = ic
                                obj["row_nums"] = list(
                                    range(obj["start_row"], obj["end_row"]+1))
                                obj["column_nums"] = list(
                                    range(obj["start_col"], obj["end_col"]+1))
                            already_seen_cell.add(cell)

                            # get chunk before in row and column
                            if ir > 0:
                                skipper_row = 1
                                while ir-skipper_row > 0 and not (area.data['cells'][ir-skipper_row][ic]):
                                    skipper_row += 1
                                prev_row = area.data['cells'][ir -
                                                              skipper_row][ic]
                                skipper_row -= 1
                                if prev_row:
                                    for prev_row_chunk, current in itertools.product(inpt.references.byId[prev_row], inpt.references.byId[cell]):
                                        content["relations"].append({
                                            # "from": prev_row_chunk.oid,
                                            # "to": cell,
                                            "from": inpt.get_area_obj(prev_row_chunk).data["content"],
                                            "to": inpt.get_area_obj(current).data["content"],
                                            "type": "column",
                                            "skips": skipper_row
                                        })
                            if ic > 0:
                                skipper_col = 1
                                while ic-skipper_col > 0 and not (area.data['cells'][ir][ic-skipper_col]):
                                    skipper_col += 1
                                prev_col = area.data['cells'][ir][ic-skipper_col]
                                skipper_col -= 1
                                if prev_col:
                                    for prev_col_chunk, current in itertools.product(inpt.references.byId[prev_col], inpt.references.byId[cell]):
                                        content["relations"].append({
                                            # "from": prev_col_chunk.oid,
                                            # "to": cell,
                                            "from": inpt.get_area_obj(prev_col_chunk).data["content"],
                                            "to": inpt.get_area_obj(current).data["content"],
                                            "type": "row",
                                            "skips": skipper_col
                                        })

                            # strings within
                            prev_chunk = None
                            for ccc in inpt.references.byId[cell]:
                                ccc = inpt.get_area_obj(ccc)
                                if not ccc:
                                    continue
                                if ccc not in already_seen_chunks:
                                    already_seen_chunks.add(ccc)
                                    content["chunks"].append({
                                        "id": ccc.oid,
                                        "pos": [ccc.boundingBox.x1, ccc.boundingBox.y1, ccc.boundingBox.x2, ccc.boundingBox.y2],
                                        "text": ccc.data['content']
                                    })
                                if prev_chunk:
                                    content["relations"].append({
                                        "from": inpt.get_area_obj(prev_chunk).data["content"],
                                        "to": ccc.data["content"],
                                        # "from": prev_chunk,
                                        # "to": ccc.oid,
                                        "type": "cell",
                                        "skips": 0
                                    })
                                prev_chunk = ccc.oid
                        content["rows"].append(current_row)

                except KeyError:
                    print("No table data found")
                    continue
                tables.append(content)
                if self.elementwise_export:
                    with open(f"{self.output_dir}/{self.prefix}{inpt.oid}_page{p_id}_table{area.oid}.{self.format}", "w") as f:
                        json.dump(content, f)
        if not self.elementwise_export:
            with open(f"{self.output_dir}/{self.prefix}{inpt.oid}.{self.format}", "w") as f:
                json.dump(tables, f)
        return inpt


class ExportTextModule(ExportModule):
    _MODULE_TYPE = "ExportTextModule"

    def __init__(self, stage: int, parameters: Optional[Dict] = {}, debug_mode: bool = False) -> None:
        if 'format' not in parameters:
            parameters['format'] = 'txt'
        super().__init__(stage, parameters, debug_mode)
        self.lines = parameters.get("lines", "Line")
        self.strings = parameters.get("strings", "String")

    def execute(self, inpt: Document) -> Document:
        # assume that string areas contain the text and are contained within a line
        # string to string whitespace
        # line to line newline
        for p_id in inpt.pages.allIds:
            with open(f"{self.output_dir}/{self.prefix}{inpt.oid}_page{p_id}.{self.format}", "w") as f:
                areas, _ = util.sort_2D_grid(
                    inpt.get_area_type(self.lines, page=p_id))
                for area in areas:
                    try:
                        for ref in sorted([inpt.get_area_obj(ref) for ref in inpt.references.byId[area.oid]], key=lambda x: x.boundingBox.x1):
                            if ref.category == self.strings:
                                f.write(ref.data['content'] + " ")
                    except KeyError:
                        print(f"No text found in {area.oid}")
                        continue
                    f.write("\n")
        return inpt


class ExportWordsModule(ExportModule):
    _MODULE_TYPE = "ExportWordsModule"

    def __init__(self, stage: int, parameters: Optional[Dict] = {}, debug_mode: bool = False) -> None:
        parameters['format'] = 'txt'
        super().__init__(stage, parameters, debug_mode)
        self.export_bbs = parameters.get("export_bbs", False)
        self.bbs_format = parameters.get("bbs_format", "image")
        self.lines = parameters.get("lines", "Line")
        self.strings = parameters.get("strings", "String")

    def execute(self, inpt: Document) -> Document:
        # assume that string areas contain the text and are contained within a line
        # string to string whitespace
        # line to line newline
        for p_id in inpt.pages.allIds:
            with open(f"{self.output_dir}/{self.prefix}{inpt.oid}_page{p_id}.{self.format}", "w") as f:
                for area in inpt.get_area_type(self.strings, page=p_id):
                    try:
                        if self.export_bbs:
                            if self.bbs_format == "image":
                                boundingBox = area.boundingBox.get_in_img_space(
                                    inpt.pages.byId[p_id].factor_width, inpt.pages.byId[p_id].factor_height)
                            else:
                                boundingBox = area.boundingBox.get_in_xml_space(
                                    inpt.pages.byId[p_id].factor_width, inpt.pages.byId[p_id].factor_height)

                            f.write(''.join(inpt.get_area_data_value(
                                area, 'content')) + '\t' + '\t'.join([str(x) for x in boundingBox.tuple()]) + "\n")
                        else:
                            f.write(
                                f"{''.join(inpt.get_area_data_value(area, 'content'))}\n")
                    except KeyError:
                        print(f"No text found in {area.oid}")
                        continue
        return inpt


class ExportImageModule(ExportModule):
    _MODULE_TYPE = "ExportImageModule"

    def __init__(self, stage: int, parameters: Optional[Dict] = {}, debug_mode: bool = False) -> None:
        if 'format' not in parameters:
            parameters['format'] = 'png'
        super().__init__(stage, parameters, debug_mode)
        self.category = parameters.get("category", "Page")

    def execute(self, inpt: Document) -> Document:
        if self.category == "Page":
            for i, p in enumerate(inpt.pages.allIds):
                inpt.get_img_page(p, as_string=False).save(
                    f"{self.output_dir}/{self.prefix}{inpt.oid}_page{i}.{self.format}")
        else:
            for i, l in enumerate(inpt.get_area_type(self.category)):
                inpt.get_img_snippet(l.oid, as_string=False).save(
                    f"{self.output_dir}/{self.prefix}{inpt.oid}_{self.category}{i}.{self.format}")
        return inpt


class ExportMaskedImageModule(ExportModule):
    _MODULE_TYPE = "ExportMaskedImageModule"

    """
    Exports pages with specified areas marked with rectangles
    """

    def __init__(self, stage: int, parameters: Optional[Dict] = {}, debug_mode: bool = False) -> None:
        if 'format' not in parameters:
            parameters['format'] = 'png'
        super().__init__(stage, parameters, debug_mode)
        self.category = parameters.get("type", ["String"])
        self.color = parameters.get("color", ["red"])
        self.thickness = parameters.get(
            "thickness", [2 for _ in range(len(self.color))])
        self.annotate_confidence = parameters.get("annotate_confidence", False)
        self.annotate_bb = parameters.get("annotate_bb", False)
        self.annotate_id = parameters.get("annotate_id", False)

        assert len(self.category) == len(
            self.color), "Number of types and colors must be equal"

    def execute(self, inpt: Document) -> Document:
        for i, p in enumerate(inpt.pages.allIds):
            img = inpt.get_img_page(p, as_string=False)
            for c_ix, t in enumerate(self.category):
                for area in inpt.get_area_type(t, page=p):
                    if area.category == "DrawnLine":
                        img = self.draw_line(inpt.pages.byId[p].factor_width,
                                             inpt.pages.byId[p].factor_width,
                                             img, area, self.color[c_ix],
                                             self.annotate_confidence,
                                             self.thickness[c_ix])
                    elif area.category == "DrawnBezier":
                        img = self.draw_bezier(inpt.pages.byId[p].factor_width,
                                               inpt.pages.byId[p].factor_width,
                                               img, area, self.color[c_ix],
                                               self.annotate_confidence,
                                               self.thickness[c_ix])
                    else:
                        img = self.draw_rectangle(inpt.pages.byId[p].factor_width,
                                                  inpt.pages.byId[p].factor_width,
                                                  img, area, self.color[c_ix],
                                                  self.annotate_confidence,
                                                  self.annotate_bb,
                                                  self.annotate_id,
                                                  self.thickness[c_ix])

            img.save(
                f"{self.output_dir}/{self.prefix}{inpt.oid}_page{i}.{self.format}")
        return inpt

    def convert_point(self, x1, y1, width_factor, height_factor):
        return x1 * width_factor, y1 * height_factor

    def draw_rectangle(self,
                       factor_width,
                       factor_height, img, area, color, annotate_confidence, annotate_bb, annotate_id, width=2):
        img_draw = ImageDraw.Draw(img)
        img_space_bb = area.boundingBox.get_in_img_space(
            factor_width, factor_height)
        img_draw.rectangle([(img_space_bb.x1, img_space_bb.y1),
                            (img_space_bb.x2, img_space_bb.y2)], outline=color, width=width)
        if annotate_confidence:
            img_draw.text((img_space_bb.x1, img_space_bb.y1),str(area.confidence), fill=color)
        if annotate_bb:
            st = f"P{round(img_space_bb.x1)},{round(img_space_bb.y1)}"
            img_draw.text((img_space_bb.x1, img_space_bb.y2), st, fill=color, font_size=4)
            st = f"P{round(img_space_bb.x2)},{round(img_space_bb.y2)}"
            img_draw.text((img_space_bb.x1, img_space_bb.y2+8), st, fill=color, font_size=4)
        if annotate_id:
            img_draw.text((img_space_bb.x1, img_space_bb.y1-10), str(area.oid), fill=color)
        return img

    def draw_line(self, factor_width, factor_height, img, area, color, annotate_confidence, width=2):
        img_draw = ImageDraw.Draw(img)

        pts = area.data.get('pts', [])
        if not pts:
            try:
                pts = area.boundingBox.tuple()
                pts = [(pts[0], pts[1]), (pts[2], pts[3])]
            except Exception:
                return img
        if not area.boundingBox.img_sp:
            p1 = self.convert_point(*pts[0], factor_width, factor_height)
            p2 = self.convert_point(*pts[1], factor_width, factor_height)
        else:
            p1 = pts[0]
            p2 = pts[1]

        img_draw.line([p1, p2], fill=color, width=width)

        if annotate_confidence:
            # half way between p1 and p2
            img_draw.text( ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2), str(area.confidence), fill=color)
        return img

    def draw_bezier(self, factor_width, factor_height, img, area, color, annotate_confidence, width=2):
        img_draw = ImageDraw.Draw(img)

        pts = area.data.get('pts')

        p1 = self.convert_point(*pts[0], factor_width, factor_height)
        p2 = self.convert_point(*pts[1], factor_width, factor_height)
        p3 = self.convert_point(*pts[2], factor_width, factor_height)
        p4 = self.convert_point(*pts[3], factor_width, factor_height)

        img_draw.line([p1, p2, p3, p4], fill=color, width=width)

        if annotate_confidence:
            img_draw.text(p1, str(area.confidence), fill=color)
        return img


class ExportMaskedSVGModule(ExportModule):
    _MODULE_TYPE = "ExportMaskedSVGModule"

    def __init__(self, stage: int, parameters: Optional[Dict] = {}, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.category = parameters.get("type", ["String"])
        self.color = parameters.get("color", ["red"])
        self.thickness = parameters.get(
            "thickness", [2 for _ in range(len(self.color))])
        self.annotate_confidence = parameters.get("annotate_confidence", False)
        self.annotate_bb = parameters.get("annotate_bb", False)

        assert len(self.category) == len(
            self.color), "Number of types and colors must be equal"

    def execute(self, inpt: Document) -> Document:
        import svgwrite.image
        import svgwrite.shapes
        for i, p in enumerate(inpt.pages.allIds):
            img = inpt.get_img_page(p, as_string=False)

            svg_root = svgwrite.Drawing(f"{self.output_dir}/{self.prefix}{inpt.oid}_page{i}.svg",
                                        size=(img.width, img.height))

            svg_root.add(svgwrite.image.Image(insert=(0, 0),
                                              size=(img.width, img.height),
                                              href=f"{self.prefix}{inpt.oid}_{p}.png"
                                              ))

            svg = self.generate_svg(inpt, svg_root, svg_root)
            svg.save()

        return inpt

    def generate_svg(self, inpt: Document, page_id: str, svg_root) -> str:
        import svgwrite

        for c_ix, t in enumerate(self.category):
            # new group
            g = svgwrite.container.Group()
            for area in inpt.get_area_type(t, page=page_id):
                if area.category == "DrawnLine":
                    g.add(self.draw_line_svg(
                        inpt.pages.byId[page_id], area, self.color[c_ix], self.thickness[c_ix]))
                elif area.category == "DrawnBezier":
                    g.add(self.draw_bezier_svg(
                        inpt.pages.byId[page_id], area, self.color[c_ix], self.thickness[c_ix]))
                else:
                    g.add(self.draw_rectangle_svg(
                        inpt.pages.byId[page_id], area, self.color[c_ix], self.thickness[c_ix]))
            svg_root.add(g)

        return svg_root

    def draw_rectangle_svg(self, page, area, color, width=2):
        import svgwrite.shapes
        img_space_bb = area.boundingBox.get_in_img_space(
            page.factor_width, page.factor_height)
        rect_attrib = {
            'insert': (img_space_bb.x1, img_space_bb.y1),
            'size': (img_space_bb.x2 - img_space_bb.x1, img_space_bb.y2 - img_space_bb.y1),
            'stroke': color,
            'stroke-width': str(width),
            'fill': 'none'
        }
        return svgwrite.shapes.Rect(**rect_attrib)

    def draw_line_svg(self, page, area, color, width=2):
        import svgwrite.shapes
        pts = area.data.get('pts', [])
        if not pts:
            try:
                pts = area.boundingBox.tuple()
                pts = [(pts[0], pts[1]), (pts[2], pts[3])]
            except Exception:
                return
        if not area.boundingBox.img_sp:
            p1 = self.convert_point(
                *pts[0], page.factor_width, page.factor_height)
            p2 = self.convert_point(
                *pts[1], page.factor_width, page.factor_height)
        else:
            p1 = pts[0]
            p2 = pts[1]

        line_attrib = {
            'x1': str(p1[0]),
            'y1': str(p1[1]),
            'x2': str(p2[0]),
            'y2': str(p2[1]),
            'stroke': color,
            'stroke-width': str(width),
        }
        return svgwrite.shapes.Line(**line_attrib)

    def draw_bezier_svg(self, page, area, color, width=2):
        import svgwrite.shapes
        pts = area.data.get('pts')

        p1 = self.convert_point(*pts[0], page.factor_width, page.factor_height)
        p2 = self.convert_point(*pts[1], page.factor_width, page.factor_height)
        p3 = self.convert_point(*pts[2], page.factor_width, page.factor_height)
        p4 = self.convert_point(*pts[3], page.factor_width, page.factor_height)

        points = f"{p1[0]},{p1[1]} {p2[0]},{p2[1]} {p3[0]},{p3[1]} {p4[0]},{p4[1]}"

        bezier_attrib = {
            'points': points,
            'stroke': color,
            'stroke-width': str(width),
            'fill': 'none'
        }
        return svgwrite.shapes.Polyline(**bezier_attrib)


class ExportCOCOModule(ExportModule):
    _MODULE_TYPE = "ExportCOCOModule"

    """
    Exports pages in COCO format
    """

    def __init__(self, stage: int, parameters: Optional[Dict] = {}, debug_mode: bool = False) -> None:
        if 'format' not in parameters:
            parameters['format'] = 'jpg'
        super().__init__(stage, parameters, debug_mode)
        self.super_categories = parameters.get("super_categories", ["Table"])
        self.type = parameters.get("type", ["Table"])

    def execute(self, inpt: Document) -> Document:
        import json

        f = {
            "info": {
                "description": "Patent Table Dataset",
                "version": "0.0",
                "year": 2024,
                "contributor": "SeKe",
                "date_created": "2024/01"
            },
            "licenses": [],
            "categories": [],
            "images": [],
            "annotations": [],
        }

        for i, p in enumerate(inpt.pages.allIds):
            img = inpt.get_img_page(p, as_string=False)

            image = dict()
            image["id"] = f"{self.prefix}{inpt.oid}_page{i}"
            image["width"] = img.width
            image["height"] = img.height
            image["file_name"] = f"{self.prefix}{inpt.oid}_page{i}.{self.format}"
            image["license"] = 0
            f["images"].append(image)

            for t, type in enumerate(self.type):
                f["categories"].append(
                    {
                        "id": t,
                        "name": self.type[t],
                        "supercategory": self.type[t]
                    }
                )

            for t, type in enumerate(self.type):
                areas = inpt.get_area_type(type, page=p)
                for area in areas:
                    img_space_bb = area.boundingBox.get_in_img_space(
                        inpt.pages.byId[p].factor_width, inpt.pages.byId[p].factor_height)
                    f["annotations"].append({
                        "id": f"{inpt.oid}-{area.oid}",
                        "image_id": f"{self.prefix}{inpt.oid}_page{i}",
                        "category_id": t,
                        "bbox": [
                            int(img_space_bb.x1),
                            int(img_space_bb.y1),
                            int(img_space_bb.width),
                            int(img_space_bb.height)
                        ],
                        "area": int(img_space_bb.width) * int(img_space_bb.height),
                        "score": area.confidence if area.confidence else 0
                    })
            img.save(
                f"{self.output_dir}/{self.prefix}{inpt.oid}_page{i}.{self.format}")

        # remove double entries in categories
        existing = set()
        for c in f["categories"].copy():
            if c["name"] in existing:
                f["categories"].remove(c)
            else:
                existing.add(c["name"])

        with open(f"{self.output_dir}/{self.prefix}{inpt.oid}.json", "w") as file:
            file.write(json.dumps(f))
        return inpt


class ExportDrawnFiguresModule(ExportModule):
    _MODULE_TYPE = "ExportDrawnFiguresModule"

    """
    Exports drawn figures
    """

    def __init__(self, stage: int, parameters: Optional[Dict] = {}, debug_mode: bool = False) -> None:
        if 'format' not in parameters:
            parameters['format'] = 'json'
        super().__init__(stage, parameters, debug_mode)
        self.apply_to = parameters.get("apply_to", "DrawnFigure")
        self.export_references = parameters.get("export_references", True)
        self.export_img = parameters.get("export_img", True)
        self.export_overlapping_text = parameters.get(
            "export_overlapping_text", True)
        self.filter_smaller_areas = parameters.get("filter_smaller_areas", 150)
        self.flat_export = parameters.get("flat_export", False)

    def execute(self, inpt: Document) -> Document:
        """
        Export in format
        [
            {
                "id": "drawn_figure_0",
                "page": 0,
                "boundingBox": [0, 0, 100, 100],
                "img": "base64",
                "objs": [
                    {
                        "id": "drawn_line_0",
                        "type": "DrawnLine",
                        "boundingBox": [0, 0, 100, 100],
                        "pts": [[0, 0], [100, 100]]
                    }
                ]
            }
        ]
        """
        export = []
        for p in inpt.pages.allIds:
            for area in inpt.get_area_type(self.apply_to, page=p):
                if area.boundingBox.area() < self.filter_smaller_areas:
                    continue
                img_space_bb = area.boundingBox.get_in_img_space(
                    inpt.pages.byId[p].factor_width, inpt.pages.byId[p].factor_height)

                try:
                    img = inpt.get_img_snippet_from_bb(
                        img_space_bb, p, as_string=True)
                except ValueError:
                    img = ""

                objs = []
                try:
                    for ref in inpt.references[area.oid]:
                        ref = inpt.get_area_obj(ref)
                        # strokes
                        add_info = ref.data

                        if not self.flat_export:
                            objs.append({
                                "id": ref.oid,
                                "type": ref.category,
                                "boundingBox": [ref.boundingBox.x1, ref.boundingBox.y1, ref.boundingBox.x2, ref.boundingBox.y2],
                                "pts": ref.data.get("pts", []),
                                "draw": add_info
                            })

                        for r in inpt.references.byId[ref.oid]:
                            r = inpt.get_area_obj(r)
                            pts = r.data.get("pts", [])
                            if not pts:
                                try:
                                    pts = r.boundingBox.tuple()
                                except Exception:
                                    continue
                            oo = {
                                "id": r.oid,
                                "type": r.category,
                                "boundingBox": [r.boundingBox.x1, r.boundingBox.y1, r.boundingBox.x2, r.boundingBox.y2],
                                "pts": pts,
                                "draw": add_info
                            }
                            if not self.flat_export:
                                try:
                                    objs[-1]['objs'].append(oo)
                                except KeyError:
                                    objs[-1]['objs'] = [oo]
                            else:
                                objs.append(oo)
                except (KeyError, TypeError):
                    continue
                if self.export_overlapping_text:
                    for text in inpt.get_area_by(lambda x: x.boundingBox.get_in_img_space(inpt.pages.byId[p].factor_width, inpt.pages.byId[p].factor_height).overlap(img_space_bb), page=p):
                        if not text.category == "String":
                            continue
                        else:
                            objs.append({
                                "id": text.oid,
                                "type": text.category,
                                "boundingBox": [text.boundingBox.x1, text.boundingBox.y1, text.boundingBox.x2, text.boundingBox.y2],
                                "content": text.data.get("content", "")
                            })

                # find closest caption
                closest = None
                closest_dist = float("inf")
                for c in inpt.get_area_type("Caption", page=p):
                    dist = c.boundingBox.get_in_img_space(inpt.pages.byId[p].factor_width, inpt.pages.byId[p].factor_height).distance(
                        area.boundingBox.get_in_img_space(inpt.pages.byId[p].factor_width, inpt.pages.byId[p].factor_height))
                    if dist < closest_dist:
                        closest = c.data.get("content", "")
                        closest_dist = dist

                export.append({
                    "id": area.oid,
                    "page": p,
                    "boundingBox": [img_space_bb.x1, img_space_bb.y1, img_space_bb.x2, img_space_bb.y2],
                    "caption": closest,
                    "img": img,
                    "objs": objs
                })
        if self.write_to_file:
            with open(f"{self.output_dir}/{self.prefix}{inpt.oid}.{self.format}", "w") as file:
                file.write(json.dumps(export))
        else:
            return export
        return inpt
