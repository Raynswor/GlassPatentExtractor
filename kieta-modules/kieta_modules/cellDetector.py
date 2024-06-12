from copy import copy
import uuid
from collections import Counter
from typing import Dict, Optional, Tuple, List, Callable, Set

from kieta_data_objs import Document, BoundingBox, Area, Font, GroupedAreas
from kieta_modules.util import range_intersect, get_overlapping_areas, group_horizontally_by_distance
from kieta_modules import Module

# from .util import  GroupedAreas

import itertools

import logging

logger = logging.getLogger('main')

class CellDetector(Module):
    _MODULE_TYPE = 'CellDetector'

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.space_between_words = float(parameters.get("space_between_words", 0))
        self.space_between_lines = float(parameters.get("space_between_lines", 0))
        self.size_difference_tolerance = float(parameters.get("size_difference_tolerance", 2))
        self.sub_super_scription_tolerance = float(parameters.get("sub/super_tolerance", 0))
        self.IoU_threshold_merge = float(parameters.get("IoU_merge_threshold", 0))

        self.apply_to = parameters.get('apply_to', 'String')
        self.lines = parameters.get('lines', 'Line')
        self.strings = parameters.get('strings', 'String')

        self.drawn_elements = parameters.get('drawn_elements', ['DrawnLine', 'DrawnRectangle', "LogicalLine"])

    def execute(self, inpt: Document) -> Document:
        l_tables = list(inpt.get_area_type('Table'))
        for table in l_tables:
            page_of_area = inpt.find_page_of_area(table)
            factors = (inpt.pages.byId[page_of_area].factor_width,
                       inpt.pages.byId[page_of_area].factor_height)
            # elements = get_overlapping_areas(table, searchable_areas, ["TableCell", "DrawnLine"])
            elements = get_overlapping_areas(table, inpt.get_area_by(
                lambda _: True, page_of_area), [self.lines, self.strings] + self.drawn_elements, factors=factors)

            text_content = None

            if self.apply_to == self.strings:
                # filter all elements without content
                # text_content = [x for x in elements[self.strings] if x.data.get('content','') != "" and x.data.get('content','') != " "]
                text_content = elements[self.strings]
            else:
                # get all text of all lines
                for l in elements[self.lines]:
                    t = list()
                    for ref in inpt.references.byId.get(l.oid, list()):
                        if inpt.areas.byId[ref].category == self.strings:
                            t.append(inpt.areas.byId[ref])
                    t.sort(key=lambda x: x.boundingBox.x1)
                    l.data['content'] = " ".join([''.join(inpt.get_area_data_value(x, 'content')) for x in t])
                text_content = elements[self.lines]

            #print(elements)
            # for e in elements["String"]:
            #    print(e.oid, e.data['content'])

            try:
                drawn_lines = [y for x in self.drawn_elements if x in elements.keys() for y in elements[x]]
            except KeyError:
                drawn_lines = []

            # get page number
            page: str = ""
            for p in inpt.pages.allIds:
                if table.oid in inpt.references.byId[p]:
                    page = p
                    break

            # detect cells
            try:
                cells: List[Area] = self.aggregate_cells(text_content, drawn_lines, inpt.fonts, (inpt.pages.byId[page].factor_width, inpt.pages.byId[page].factor_height))
                for c in cells:
                    inpt.add_area(page, category="TableCell",
                                   boundingBox=c.boundingBox,
                                   data={'content': c.data.get('content','')},
                                   referenced_by=table.oid,
                                   references=c.data['consist'],
                                   convert_to_xml=not c.boundingBox.img_sp)
            except KeyError as e:
                logger.error(e, "didn't find xml element strings")

        return inpt


    def aggregate_cells(self, flat_elements: List[Area], drawn_lines: List[Area], fonts: List[Font], factors) -> List[Area]:
        # change drawn_lines to list of boundingBoxes in img space
        drawn_lines = [x.boundingBox.get_in_img_space(*factors) for x in drawn_lines]

        res_elements = self.create_cells(flat_elements, drawn_lines, self.horizontal_cells, fonts, factors)
        # res_elements = self.create_cells(new_elements, drawn_lines, self.vertical_cells, fonts)
        test = ""
        for x in res_elements:
            test += x.data['content'] + "|||"
        # print(test)
        # check if cells overlap or are within each other
        # then merge them
        # do until no changes are made
        # logger.debug(f"cells before merging: {len(res_elements)}")
        # while True:
        #     is_within = list()
        #     for i, el in enumerate(res_elements):
        #         for j, ne in enumerate(res_elements):
        #             if i != j:
        #                 if el.boundingBox in ne.boundingBox or ne.boundingBox in el.boundingBox:
        #                     is_within.append((el, ne))
        #                     break
        #     if len(is_within) == 0:
        #         break
        #     else:
        #         res_elements = self.merge_cells(res_elements, is_within)
        # logger.debug(f"cells after merging: {len(res_elements)}")
        return res_elements

    def guess_space(self):
        # primitive_rows: List[GroupedAreas] = group_horizontally_by_distance(flat_elements, 10000, 3, 2)

        # distances_between_rows = [t for row1, row2 in itertools.pairwise(primitive_rows) if (t := row2.get_boundingBox().distance_vertical(row1.get_boundingBox())) < 50]
        # # distances_between_rows = [row2.get_boundingBox().distance_vertical(row1.get_boundingBox()) for row1, row2 in itertools.pairwise(primitive_rows)]
        # distances_between_rows_count = Counter(distances_between_rows)
        # # plot this
        # # import matplotlib.pyplot as plt
        # # plt.plot(distances_between_rows)
        # # plt.show()

        # # logger.debug(distances_between_rows_count)
        # try:
        #     # most frequent non-zero value
        #     space_between_lines =  max(distances_between_rows_count.items(), key=lambda x: x[1])[0]
        # except IndexError:
        #     space_between_lines = 6

        # distances_within_row = [item_2.boundingBox.distance_horizontal(item_1.boundingBox) for x in primitive_rows for item_1, item_2 in itertools.pairwise(sorted(x.areas, key=lambda y: y.boundingBox.x1))]
        # # self.debug_msg(Counter(distances_within_row))
        # try:
        #     space_between_words = max(Counter(distances_within_row).items(), key=lambda x: x[1])[0]
        # except IndexError:
        #     space_between_words = 6

        # # plt.plot(distances_within_row)
        # # plt.show()

        # self.info_msg(f"space between words: {space_between_words} from {self.space_between_words}")
        # self.info_msg(f"space between lines: {space_between_lines} from {self.space_between_lines}")
        # self.space_between_words = space_between_words
        # self.space_between_lines = space_between_lines

        # new_elements = self.create_cells(flat_elements, drawn_lines, self.horizontal_cells, fonts)
        pass

    def merge_cells(self, res_elements: List[Area], is_within: List[Tuple[Area, Area]]):
        for el, ne in is_within:
            # check which one is to the left
            if el.boundingBox.x1 < ne.boundingBox.x1:
                merger = el
                mergee = ne
            else:
                merger = ne
                mergee = el
            delet = -1
            for ix, i in enumerate(res_elements):
                if i.oid == mergee.oid:
                    delet = ix
                    break
            if delet != -1:
                del res_elements[delet]
            merger.boundingBox.expand(mergee.boundingBox)
            merger.data['content'] += mergee.data['content']
            merger.data['consist'].append(mergee.oid)
        return res_elements

    def horizontal_cells(self, bb: BoundingBox, nb: BoundingBox, drawn_lines: List[BoundingBox], fonts: List[Font]):
        # check any vertical overlap
        if bb.overlap_vertically(nb):
            # check if horizontal space is leq than size of 1.5 whitespace
            # with open("log.txt", "a") as f:
            #     f.write(f"{el.data['content']} - {bb.x1} {bb.x2} {nb.x1} {nb.x2}\n")
            #     f.write(f"{ne.data['content']} - {bb.y1} {bb.y2} {nb.y1} {nb.y2}\n")
            #     f.write(f"{bb.distance_horizontal(nb)}\n")
            if bb.distance_horizontal(nb) <= self.space_between_words:
                # print(bb.distance_horizontal(nb))
                # check if sub- or superscript, attach automatically
                # font = [x for x in fonts if x.oid == el.data['font']][0]
                # if (font.is_sub_script() or font.is_super_script()) and \
                #         (max(nb.y2, bb.y2) - min(nb.y1, bb.y1)) < (ne.boundingBox.y2 + el.boundingBox.y2):
                #     return True, True
                # # check if approx on the same y level
                # if abs(bb.y1 - nb.y1) < (temp := (ne.boundingBox.y2 + el.boundingBox.y2) * 0.5) and abs(
                #     bb.y2 - nb.y2) < temp:
                if (abs(bb.y1 - nb.y1) < self.size_difference_tolerance and abs(bb.y2 - nb.y2) < self.size_difference_tolerance) or  \
                (bb.y1 > nb.y1 and bb.y2 < nb.y2) or (bb.y1 < nb.y1 and bb.y2 > nb.y2):  # one is within the other
                    new_bb = bb.__expand__(nb)
                    for line in drawn_lines:  # check if overlap with line
                        # NOTE: DOES NOT WORK ON INCORRECTLY CONSTRUCTED TABLES
                        if line.intersects(new_bb) and not new_bb in line:
                            # check if new_bb is fully within line
                            return False, False
                    return True, False
                return False, False
    def vertical_cells(self, bb: BoundingBox, nb: BoundingBox, drawn_lines: List[Area], fonts: List[Font]):
        # check any vertical overlap
        if range_intersect((bb.x1, bb.x2), (nb.x1, nb.x2)):
            # check if vertical space is less than (20%) most frequent row space
            if (abs(max(bb.y1, nb.y1) - min(bb.y2, nb.y2)) < self.space_between_lines):
                new_bb = bb.__expand__(nb)
                for line in drawn_lines:  # check if there is overlap with line
                    if line.overlap(new_bb):
                        return False, False
                return True, False
            return False, False

    def create_cells(self, initial_elements: List[Area],
                     drawn_lines: List[Area],
                     fkt: Callable[[Area, Area, BoundingBox, BoundingBox], Tuple[int, bool]],
                     fonts, factors):
        new_elements: List[Area] = list()
        for el in initial_elements:
            merge_with = -1

            is_super_or_subscript = False
            for ne_ix, ne in enumerate(new_elements):
                try:
                    if (t := fkt(el.boundingBox.get_in_img_space(*factors), ne.boundingBox.get_in_img_space(*factors), drawn_lines, fonts)) and t[0]:
                        merge_with, is_super_or_subscript = ne_ix, t[1]
                        break
                except Exception as e:
                    self.debug_msg(e)
                    continue
            if merge_with < 0:
                element = el
                content = element.data.get('content', '')
                consist = [element.oid] if 'consist' not in element.data.keys() else element.data['consist']
            else:
                element = new_elements.pop(merge_with)
                content = " ".join([el.data.get('content',''), element.data.get('content','')]) \
                    if el.boundingBox.x1 < element.boundingBox.x1 \
                    else " ".join([element.data.get('content',''), el.data.get('content','')])
                try:
                    if 'consist' not in element.data.keys():
                        consist = el.data["consist"] + [element.oid]
                    else:
                        consist = el.data["consist"] + element.data['consist']
                except KeyError:
                    if 'consist' not in element.data.keys():
                        consist = [el.oid, element.oid]
                    else:
                        consist = [el.oid] + element.data['consist']
            # if is_super_or_subscript:
            #     bounds = BoundingBox(min(element.boundingBox.x1, el.boundingBox.x1),
            #                          element.boundingBox.y1,
            #                          max(element.boundingBox.x2, el.boundingBox.x2),
            #                          element.boundingBox.y2)
            # else:
            # bounds = element.boundingBox.__expand__(el.boundingBox)
            bounds = BoundingBox(min(element.boundingBox.x1, el.boundingBox.x1),
                                 min(element.boundingBox.y1, el.boundingBox.y1),
                                 max(element.boundingBox.x2, el.boundingBox.x2),
                                 max(element.boundingBox.y2, el.boundingBox.y2),
                                 img_sp=element.boundingBox.img_sp)
            new_elements.append(Area(
                oid=uuid.uuid4().hex[:10],
                boundingBox=bounds,
                category="TableCell",
                data={"content": content, "consist": consist})
            )
        return new_elements
