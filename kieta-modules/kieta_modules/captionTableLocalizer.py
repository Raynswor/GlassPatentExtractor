import itertools
import math
import re
from os.path import basename
from typing import Dict, Optional, List

import numpy as np
from kieta_data_objs import Document, BoundingBox, Area, Page
from kieta_modules import Module
from kieta_modules.util import group_horizontally_by_distance, get_overlapping_areas

import logging
logger = logging.getLogger('main')

def in_same_page_half(page_width, a: Area, b: Area) -> bool:
    page_middle = page_width*0.5

    a_middle = a.boundingBox.x1 + (a.boundingBox.x2 - a.boundingBox.x1)*0.5
    b_middle = b.boundingBox.x1 + (b.boundingBox.x2 - b.boundingBox.x1)*0.5


    # print(f"page_middle: {page_middle}, "
    #       f"{a.data['content']}: {a_middle} {a.boundingBox}, "
    #       f"{b.data['content']}: {b_middle} {b.boundingBox}")
    if a_middle < page_middle and b_middle < page_middle:
        return True
    elif a_middle >= page_middle and b_middle >= page_middle:
        return True
    elif a_middle < page_middle and b.boundingBox.x1 < page_middle and b.boundingBox.x2 > page_middle:
        return True
    else:
        return False


class CaptionTableLocalizer(Module):
    _MODULE_TYPE = 'CaptionTableLocalizer'

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.threshold: int = int(parameters["minimum_distance_line_caption"])
        self.table_direction: str = parameters["table_direction"]
        self.minimum_width_line: int = 150

    def execute(self, doc: Document) -> Document:
        # get all captions
        captions = list(doc.get_area_type('Caption'))
        for ix, c in enumerate(captions):
            page = f"page-{c.oid.split('-')[0]}"

            matches = re.findall(r"\d+", c.data['content'])
            table_no = ix + 1
            if len(matches) > 0:
                table_no = matches[0]
            else:
                logger.info('did not find table number, using table caption index')
            logger.debug(f'TABLE NUMBER {table_no}')

            # get every area that intersects vertically with table caption
            search_space: List[Area] = doc.get_area_by(lambda x: x.oid!=c.oid and not x.category in {"Line","Block"} and not doc.is_referenced_by(x.oid, c.oid))
            
            # [a for a in doc.references.byId[page] if
            #                             c.oid != a and doc.areas.byId[a].category != "Line" and a not in doc.references.byId[c.oid]]


            # shrink search space, if above only above, if below only below
            if self.table_direction == 'above':
                search_space = list(filter(lambda x: x.boundingBox.y1 < c.boundingBox.y1, search_space))
                search_space.sort(key=lambda x: x.boundingBox.y1, reverse=True)
            elif self.table_direction == 'below':
                search_space = list(filter(lambda x: x.boundingBox.y2 >= c.boundingBox.y2, search_space))
                search_space.sort(key=lambda x: x.boundingBox.y1)

            other_caption_id: Area = ""
            for t in search_space:  # find lowest other table caption above
                if not t.category == "Caption":
                    continue
                if not in_same_page_half(doc.pages.byId[f"page-{t.oid[0]}"].img_width,  c, t):
                    continue
                if other_caption_id == "":
                    other_caption_id = t
                    continue
                if (self.table_direction == 'above' and other_caption_id.boundingBox.y1 < t.boundingBox.y1) or \
                (self.table_direction == 'below' and other_caption_id.boundingBox.y1 > t.boundingBox.y1):
                    # check if
                    other_caption_id = t

            # snip before other caption
            if other_caption_id:
                if self.table_direction == 'above':
                    search_space = list(filter(lambda x: x.boundingBox.y1 > other_caption_id.boundingBox.y2 - 5,search_space))
                elif self.table_direction == 'below':
                    search_space = list(filter(lambda x: x.boundingBox.y2 < other_caption_id.boundingBox.y1 + 5,search_space))

            cached_search_space = list(filter(lambda x: x.category == "String", search_space))

            # horizontal lines in search space overlapping with caption
            # print(list([(doc.areas.byId[p].boundingBox.width,doc.areas.byId[p].boundingBox.height) for p in search_space if doc.areas.byId[p].category == "DrawnLine"]))
            horizontal_lines = sorted(filter(lambda x: x.category == "DrawnLine" and
                                                       x.boundingBox.is_horizontal() and
                                                       c.boundingBox.overlap_horizontally(x.boundingBox) and
                                                        x.boundingBox.width > self.minimum_width_line,
                                                     search_space),
                                              key=lambda x: x.boundingBox.y1)
            logger.debug(f"horizontal lines {len(horizontal_lines)}, {horizontal_lines}")
            if len(horizontal_lines) == 0:
                logger.debug('no horizontal lines found, use distance based approach, NOT IMPLEMENTED YET')
            if len(horizontal_lines) >= 1:  # snip vertically at longest line
                # find longest line
                longest_line = [(x, x.boundingBox.width) for x in horizontal_lines]
                longest_line = sorted(longest_line, key=lambda x: x[1], reverse=True)[0][0]
                # snip at x1 and x2 of longest line
                logger.debug(f"elements before snipping {len(search_space)}")
                search_space = list(filter(
                    lambda x: x.boundingBox.x1 >= longest_line.boundingBox.x1 - 10 and
                              x.boundingBox.x2 <= longest_line.boundingBox.x2 + 10,
                    search_space))
                logger.debug(f"elements after snipping {len(search_space)}")
            logger.debug([i for i in horizontal_lines])
            if len(horizontal_lines) >= 2:
                # find lines with biggest distance
                distances = list()
                # calculate distance between all lines
                for (k,v) in itertools.product([horizontal_lines[0]], horizontal_lines):
                    if k == v:
                        continue
                    # compared lines should have almost the same length 90% - 110%
                    if k.boundingBo * 0.8 <= v.boundingBox.width <= k.boundingBox.width * 1.2:
                        distances.append((k, v, k.boundingBox.distance(v.boundingBox)))
                distances.sort(key=lambda x: x[2], reverse=True)

                # if distance between these lines is big enough, cut off at the farthest line
                if distances and distances[0][2] >= self.threshold:
                    # get topline and bottomline
                    # remove all elements that are not between these lines
                    topline = distances[0][0] if distances[0][0].boundingBox.y1 < distances[0][1].boundingBox.y1 else distances[0][1]
                    bottomline = distances[0][1] if topline == distances[0][0] else distances[0][0]

                    # print(f"topline {topline}, bottomline: {bottomline}")
                    # print(f"len before {len(search_space)}")
                    search_space = list(filter(lambda x: x.boundingBox.y1 > topline.boundingBox.y1-10 and
                    x.boundingBox.y1 < bottomline.boundingBox.y1+10, search_space))
                    # print(f"len after {len(search_space)}")
                    logger.debug(f"elements between lines {len(search_space)}")

            bb = BoundingBox(10000, 10000, 0, 0)
            for x in search_space:
                bb.expand(x.boundingBox)
            # extend search space again if the number of elements to be added is small
            # extend by 10% of current box height
            groups = group_horizontally_by_distance( get_overlapping_areas(bb, list([doc.get_area_obj(x) for x in doc.references.byId[page]]), ['Line'])['Line'], 1000, 1, 5)
            # groups = group_horizontally_by_distance([doc.areas.byId[x] for x in search_space], 1000, 1, 5)
            # debug_plot_bb(doc, page, [g.get_boundingBox() for g in groups])
            average_bounding_box_height = np.average([x.boundingBox.height for x in list(filter(lambda x: x.category == "String", search_space))])

            try:
                max_elements_in_groups = max([len(x) for x in groups])
            except ValueError:
                max_elements_in_groups = 0
            logger.debug(f"avg elements per row {max_elements_in_groups}, average_bounding_box_height {average_bounding_box_height}")

            # expansions = [bb.__expand__(BoundingBox(10000,10000,0,0))]

            empty_expansions = 0
            while bb.y2 < doc.pages.byId[page].img_height and empty_expansions < 3:
                current_no_elements = len(list(filter(lambda x: x.category == "String", search_space)))
                if current_no_elements == len(cached_search_space):
                    break
                expanded_elements = list()

                if self.table_direction == "below":
                    bb.expand_by(0, 0, 0, average_bounding_box_height)
                    expanded_elements = get_overlapping_areas(bb, list([x for x in cached_search_space]), ['String'])['String']
                elif self.table_direction == "above":
                    pass
                # expansions.append(bb.__expand__(BoundingBox(10000,10000,0,0)))
                logger.debug(f"expanded search space {bb.y2} -> {bb.y2 + average_bounding_box_height} to {len(expanded_elements)} elements from {current_no_elements} elements")
                if len(expanded_elements) - current_no_elements < max_elements_in_groups*2:
                    # logger.debug("did that")
                    if len(expanded_elements) == len(search_space):
                        empty_expansions += 1
                    else:
                        empty_expansions = 0
                    search_space = [x.oid for x in expanded_elements]
                else:
                    break
            # debug_plot_bb(doc, page, expansions, "expanded")
            bb = BoundingBox(10000, 10000, 0, 0)
            for x in search_space:
                bb.expand(x.boundingBox)

            if doc.pages.byId[page].layout.rotation == 90 or doc.pages.byId[page].layout.rotation == 270:
                bb.transpose()
            doc.add_area(page, "Table", bb, data={'number': table_no})
            # # other captions are cutoff points
            # # einschraenkung, dass gleiche seite
        return doc

    def filter_search_area(self, tokens: List, lmbds: List, current_bb: BoundingBox, other_bbs: List[BoundingBox],
                           rev=False) -> List:
        filtered_tokens = list()

        others = sorted(
            [dbb for dbb in other_bbs if lmbds[0](current_bb, dbb)],
            key=lmbds[1], reverse=rev)

        if others:
            for (t, bb) in tokens:
                if lmbds[2](bb, others[0]):
                    filtered_tokens.append((t, bb))
            return filtered_tokens
        else:
            return tokens
