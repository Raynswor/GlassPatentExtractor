from collections import defaultdict
import itertools
from collections import Counter, defaultdict

from kieta_data_objs import Area, BoundingBox, GroupedAreas
from typing import Callable, List, Tuple, Optional, Dict, Iterable, Any, Union

import logging

import networkx as nx

from kieta_data_objs import Document

from numbers import Number

import numpy as np

logger = logging.getLogger('main')


class UnionFind:

    @staticmethod
    def from_count(element_count: int):
        return UnionFind(range(element_count))

    @staticmethod
    def from_max_label(max_label: int):
        return UnionFind.from_count(int(max_label) + 1)

    def __init__(self, iterable):  # e.g. use range(100) for 100 nodes
        self.tree = list(iterable)

    def __iter__(self):
        return iter(self.tree)

    def find(self, element: int) -> int:
        tree = self.tree
        cur = element
        if tree[cur] == cur:
            return cur

        # find parent
        while True:
            cur = tree[cur]
            if tree[cur] == cur:
                break
        parent = cur

        # compress
        cur = element
        while tree[cur] != parent:
            tree[cur], cur = parent, tree[cur]
        return parent

    def union2(self, element1: int, element2: int):
        par1 = self.find(element1)
        par2 = self.find(element2)
        self.tree[par2] = par1
        return self

    def union_iterable(self, iterable):
        for e1, e2 in iterable:
            self.union2(e1, e2)
        return self

    # returns a dict of lists
    # each item in the dict is a connected component, the list elements
    # describe the indices of all the elements in that CC
    def get_sets(self) -> Dict[int, List[int]]:
        ccs = defaultdict(list)
        for id, _ in enumerate(self.tree):
            parent = self.find(id)
            ccs[parent].append(id)

        return ccs


def find_overlapping_entries(x1: Number, x2: Number, a1: np.ndarray, a2: np.ndarray, window_size: int = 0) -> np.ndarray:
    """
    Find overlapping entries in two arrays of intervals

    Parameters
    ----------
    x1 : number
        Start of the search interval
    x2 : number
        End of the search interval
    a1 : array
        Start of the intervals
    a2 : array
        End of the intervals
    window_size : number, optional
        The size of the window around the intervals, by default 0

    Returns
    -------
    array
        Mask of overlapping intervals
    """
    # Create a mask for entries that overlap
    mask = (x1 <= a2 + window_size) & (x2 >= a1 - window_size)
    return mask


def find_overlapping_entries_2d(search_index: int,
                                all_elements: np.ndarray,
                                window_size_main: int = 0,
                                window_size_other: int = 0,
                                excluded_entries: np.ndarray = [],
                                direction: int = 0,
                                height_threshold: float = 0,
                                ) -> np.ndarray:
    x1, y1, x2, y2 = all_elements[search_index]

    if direction == 0:  # horizontal
        mask = find_overlapping_entries(
            x1, x2, all_elements[:, 0], all_elements[:, 2], window_size_main)
    else:  # vertical
        mask = find_overlapping_entries(
            y1, y2, all_elements[:, 1], all_elements[:, 3], window_size_main)

    if window_size_other:
        if direction == 0:
            mask = mask & find_overlapping_entries(
                y1, y2, all_elements[:, 1], all_elements[:, 3], window_size_other)
        else:
            mask = mask & find_overlapping_entries(
                x1, x2, all_elements[:, 0], all_elements[:, 2], window_size_other)

    # mask[search_index] = False

    mask = np.where(mask)[0]

    if len(mask) > 0:
        for idx in mask:
            # height threshold
            new_box_y1 = min(y1, all_elements[idx, 1])
            new_box_y2 = max(y2, all_elements[idx, 3])

            if height_threshold and (new_box_y2 - new_box_y1) > height_threshold:
                # remove from mask
                mask = np.delete(mask, np.where(mask == idx))
                continue

            if len(excluded_entries) > 0:
                # intersection
                new_box_x1 = min(x1, all_elements[idx, 0])
                new_box_x2 = max(x2, all_elements[idx, 2])

                lines_overlap = np.where(
                    np.logical_or(
                        np.logical_and(
                            np.logical_and(
                                new_box_x1 < excluded_entries[:, 0],
                                new_box_x2 > excluded_entries[:, 0]
                            ),
                            np.logical_and(
                                new_box_y1 > excluded_entries[:, 1],
                                new_box_y2 < excluded_entries[:, 3])
                        ),
                        np.logical_and(
                            np.logical_and(
                                new_box_y1 < excluded_entries[:, 1],
                                new_box_y2 > excluded_entries[:, 1]
                            ),
                            np.logical_and(
                                new_box_x1 > excluded_entries[:, 0],
                                new_box_x2 < excluded_entries[:, 2])
                        )
                    )
                )[0]
                if len(lines_overlap) > 0:
                    mask = np.delete(mask, np.where(mask == idx))
                    continue

    return mask


def merge_overlapping_bounding_boxes_in_one_direction(
        boxes: List[BoundingBox],
        window_size_main,
        window_size_other,
        excluded_entries: Union[List[BoundingBox], np.ndarray] = [],
        direction: int = 0,
        height_threshold: int = 0) -> Tuple[List[BoundingBox], List[List]]:
    """
    Is effectively non-maximum suppression for bounding boxes in one direction

    window_size_other: int = 0: window size for the other direction, + increases tolerance, - decreases tolerance
    direction: int = 0: 0=horizontal, 1=vertical
    
    return a list of merged bounding boxes and merge hierarchy
    """
    
    space = boxes[0].img_sp

    all_bbs = np.array([tuple(i) for i in boxes])
    # if exclusion not an ndarray, convert to ndarray
    if not isinstance(excluded_entries, np.ndarray):
        excluded_entries = np.array([tuple(i) for i in excluded_entries])

    uf = UnionFind.from_count(len(boxes))
    for i in range(len(all_bbs)):
        for idx in find_overlapping_entries_2d(search_index=i, all_elements=all_bbs, excluded_entries=excluded_entries,
                                                      window_size_main=window_size_main, window_size_other=window_size_other, direction=direction, height_threshold=height_threshold):
            uf.union2(i, idx)

    res = list()

    for pop_idx in uf.get_sets().values():
        x1 = np.min(all_bbs[pop_idx, 0])
        y1 = np.min(all_bbs[pop_idx, 1])
        x2 = np.max(all_bbs[pop_idx, 2])
        y2 = np.max(all_bbs[pop_idx, 3])

        bb = BoundingBox(x1, y1, x2, y2, img_sp=space)
        res.append(bb)
    
    return res, list(uf.get_sets().values())
    


def group_horizontally_by_distance(areas: List[Area], threshold_horizontal: int, threshold_height_diff: int, threshold_vertical_translation: int) -> List[GroupedAreas]:
    # Initialize empty dict to store groups
    groups: List[GroupedAreas] = []
    # Iterate over each object
    for obj in areas:
        done = False
        # ignore objects that are higher than wide
        if (t := obj.boundingBox.y2-obj.boundingBox.y1) > obj.boundingBox.x2-obj.boundingBox.x1 and t > 30:
            continue

        # Iterate over other objects
        for other_group in groups:
            for other in other_group.areas:
                if obj.oid == other.oid or other.boundingBox.y2-other.boundingBox.y1 > other.boundingBox.x2-other.boundingBox.x1:
                    continue
                # check if same vertical position
                if abs(obj.boundingBox.y1-threshold_vertical_translation <= other.boundingBox.y1 <= obj.boundingBox.y1+threshold_vertical_translation) and \
                   abs(obj.boundingBox.y2-threshold_vertical_translation <= other.boundingBox.y2 <= obj.boundingBox.y2+threshold_vertical_translation):
                    # Calculate horizontal distance between objects
                    if abs(obj.boundingBox.x1 - other.boundingBox.x2) < threshold_horizontal or abs(obj.boundingBox.x2 - other.boundingBox.x1) < threshold_horizontal:
                        # calculate vertical translation
                        if abs((obj.boundingBox.y2-obj.boundingBox.y1)-(other.boundingBox.y2-other.boundingBox.y1)) < threshold_height_diff:
                            other_group.areas.append(obj)
                            done = True
                            break
            if done:
                break
        if not done:
            groups.append(GroupedAreas([obj]))
    groups.sort(key=lambda x: x.get_boundingBox().y1)

    # check if there are groups that are contained in other groups
    changed = True
    logger.debug(f"before merging: {len(groups)}")
    while changed:
        changed = False
        for (i, j) in itertools.product(range(len(groups)), repeat=2):
            if i == j:
                continue
            if groups[i].get_boundingBox().intersection_over_union(groups[j].get_boundingBox()) > 0.1 or \
                groups[i].get_boundingBox() in groups[j].get_boundingBox() or \
                    groups[j].get_boundingBox() in groups[i].get_boundingBox():
                groups[j].areas.extend(groups[i].areas)
                del groups[i]
                changed = True
                break

    return groups


def sort_into_two_lists(l: List, condition: Callable) -> Tuple[List, List]:
    """
    Sort a list into two lists based on a condition

    Parameters
    ----------
    l : list
        The list to sort
    condition : function
        The condition to sort by

    Returns
    -------
    tuple
        Two lists, one with elements that satisfy the condition and one with elements that do not
    """
    one, two = list(), list()
    for x in l:
        if condition(x):
            one.append(x)
        else:
            two.append(x)
    return one, two


def get_overlapping_areas(area: Union[Area, BoundingBox],
                          areas: List[Area],
                          filter_categories: Optional[List[str]],
                          factors: Optional[Tuple[float, float]] = None) -> Dict[str, List[Area]]:
    """
    Get all areas that overlap with `area`, grouped by categories
    """
    if not factors:
        raise ValueError("factors must be provided")

    area = Area("", "", area) if isinstance(area, BoundingBox) else area
    bb = area.boundingBox.get_in_img_space(
        *factors) if not area.boundingBox.img_sp else area.boundingBox

    res = defaultdict(list)
    for a in areas:
        if a.oid == area.oid:
            continue

        o_bb = a.boundingBox.get_in_img_space(
            *factors) if not a.boundingBox.img_sp else a.boundingBox

        assert bb.img_sp == o_bb.img_sp

        if bb.overlap(o_bb):
            res[a.category].append(a)

    return {f: res.get(f, []) for f in filter_categories} if filter_categories else dict(res)


def group_vertically_by_alignment(lines: List[GroupedAreas], tolerance) -> List[Tuple[float, float, float]]:
    rets: List[GroupedAreas] = []
    for l in lines:
        for a in l.areas:
            added = False
            for r in rets:
                # check if there is alignment
                if a.boundingBox.x1-tolerance <= r.get_boundingBox().x1 <= a.boundingBox.x1+tolerance or \
                   a.boundingBox.x2-tolerance <= r.get_boundingBox().x2 <= a.boundingBox.x2+tolerance or \
                   0.5*(a.boundingBox.x1+a.boundingBox.x2)-tolerance <= 0.5*(r.get_boundingBox().x1+r.get_boundingBox().x2) <= 0.5*(a.boundingBox.x1+a.boundingBox.x2)+tolerance:
                    added = True
                    r.areas.append(a)
                    break
            if not added:
                rets.append(GroupedAreas([a]))
    return rets


def range_intersect(r1: Tuple[float, float], r2: Tuple[float, float]):
    rr1 = range(int(r1[0]), int(r1[1]))
    rr2 = range(int(r2[0]), int(r2[1]))
    return range(max(rr1.start, rr2.start), min(rr1.stop, rr2.stop)) or None


def get_next_smaller(x: int, numbers: Iterable[int]) -> int:
    next_smaller_number = 0
    # Iterate through the list of numbers
    for num in numbers:
        # Check if the given number is greater than the current number
        # and the current number is greater than the next smaller number
        if x > num > next_smaller_number:
            # Update the next smaller number
            next_smaller_number = num
    return next_smaller_number


def get_most_frequent_value(ls: Iterable[Any]) -> Tuple[Any, int]:
    return Counter(ls).most_common(1)[0]


def nms_delete(boxes: List[BoundingBox], threshold) -> List[BoundingBox]:
    import numpy as np

    if len(boxes) == 0:
        return []

    pick = []

    # coordinates of bounding boxes
    boxes = np.array([b.tuple() for b in boxes])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.array(range(len(boxes)))

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index value to
        # the list of picked indexes, then initialize the suppression list
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the
        # smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap between the computed
        # bounding box and the bounding box in the area list
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(
            ([last], np.where(overlap > threshold)[0])))

    # return only the bounding boxes that were picked
    return [BoundingBox(*x) for x in boxes[pick].astype("int")]


def debug_print_table(document: Document, table: Union[str, Area], show_oid: bool = False) -> List[List[str]]:
    # get page of table_id
    table = document.areas.byId[table] if isinstance(table, str) else table
    tab_content = []
    try:
        for ir, row in enumerate(table.data['cells']):
            string_row = []
            for cell in row:
                if not cell:
                    continue
                area_obj = document.get_area_obj(cell)
                st = area_obj.data['content']
                if show_oid:
                    st += f"{area_obj.oid})"
                if area_obj.data.get("row_label", None):
                    st = "RL<" + st
                if area_obj.data.get("column_label", None):
                    st = "CL<" + st
                string_row.append(st)
            tab_content.append(string_row)
    except KeyError:
        print("No table data found")
        return

    import tabulate
    print(tabulate.tabulate(tab_content, tablefmt="grid"))
    return tab_content


def nms_delete(boxes: List[BoundingBox], threshold) -> List[BoundingBox]:
    import numpy as np

    if len(boxes) == 0:
        return []

    pick = []

    # coordinates of bounding boxes
    boxes = np.array([b.tuple() for b in boxes])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.array(range(len(boxes)))

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index value to
        # the list of picked indexes, then initialize the suppression list
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the
        # smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap between the computed
        # bounding box and the bounding box in the area list
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(
            ([last], np.where(overlap > threshold)[0])))

    # return only the bounding boxes that were picked
    return [BoundingBox(*x) for x in boxes[pick].astype("int")]


def nms_merge(boxes: List[BoundingBox], threshold, scores: List = []) -> List[BoundingBox]:
    import numpy as np

    if len(boxes) == 0:
        return []
    
    img_sp = boxes[0].img_sp

    # coordinates of bounding boxes
    boxes = np.array([b.tuple() for b in boxes])
    final_boxes = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    if len(scores) > 0:
        scores = np.array(scores)
        finale_scores = list()

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.array(range(len(boxes)))

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index value to
        # the list of picked indexes, then initialize the suppression list
        last = len(idxs) - 1
        i = idxs[last]
        merge_list = [i]

        # find the largest (x, y) coordinates for the start of the bounding box and the
        # smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap between the computed
        # bounding box and the bounding box in the area list
        overlap = (w * h) / area[idxs[:last]]

        to_merge = np.where(overlap > threshold)[0]
        merge_list.extend(idxs[to_merge])

        # check if a bounding box is completely within another bounding box
        to_remove = []
        for j in merge_list:
            if i != j and x1[i] <= x1[j] and y1[i] <= y1[j] and x2[i] >= x2[j] and y2[i] >= y2[j]:
                to_remove.append(j)
        merge_list = [j for j in merge_list if j not in to_remove]

        if len(scores) > 0:
            # compute the score of the bounding box
            finale_scores.append(np.mean(scores[merge_list]))

        # merge all boxes in merge_list
        merged_box = [float(np.min(x1[merge_list])), float(np.min(y1[merge_list])),
                      float(np.max(x2[merge_list])), float(np.max(y2[merge_list]))]
        final_boxes.append(BoundingBox(*merged_box, img_sp=img_sp))

        # delete all indexes from the index list that were merged
        idxs = np.delete(idxs, np.concatenate(([last], to_merge)))

    # return only the bounding boxes that were picked
    if len(scores) > 0:
        return final_boxes, finale_scores
    else:
        return final_boxes


def nms_merge_with_index(boxes: List[BoundingBox], threshold, scores: List = []) -> List[BoundingBox]:
    import numpy as np

    if len(boxes) == 0:
        return [], {}

    # coordinates of bounding boxes
    boxes = np.array([b.tuple() for b in boxes])
    final_boxes = []
    consists_of = {k: [] for k in range(len(boxes))}

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    if len(scores) > 0:
        scores = np.array(scores)
        finale_scores = list()

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.array(range(len(boxes)))

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index value to
        # the list of picked indexes, then initialize the suppression list
        last = len(idxs) - 1
        i = idxs[last]
        merge_list = [i]

        # find the largest (x, y) coordinates for the start of the bounding box and the
        # smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap between the computed
        # bounding box and the bounding box in the area list
        overlap = (w * h) / area[idxs[:last]]
        # print(overlap)
        to_merge = np.where(overlap > threshold)[0]

        merge_list.extend(idxs[to_merge])

        # check if a bounding box is completely within another bounding box
        to_remove = []
        for j in merge_list:
            if i != j and x1[i] <= x1[j] and y1[i] <= y1[j] and x2[i] >= x2[j] and y2[i] >= y2[j]:
                to_remove.append(j)
        merge_list = [j for j in merge_list if j not in to_remove]

        if len(scores) > 0:
            # compute the score of the bounding box
            finale_scores.append(round(np.mean(scores[merge_list]), 2))

        # merge all boxes in merge_list
        merged_box = [float(np.min(x1[merge_list])), float(np.min(y1[merge_list])),
                      float(np.max(x2[merge_list])), float(np.max(y2[merge_list]))]
        final_boxes.append(BoundingBox(*merged_box))

        consists_of[last] = merge_list

        # delete all indexes from the index list that were merged
        idxs = np.delete(idxs, np.concatenate(([last], to_merge)))

    # return only the bounding boxes that were picked
    if len(scores) > 0:
        return final_boxes, consists_of, finale_scores
    else:
        return final_boxes, consists_of


def convert_table_to_graph(doc, table_area: Union[str, Area]) -> nx.Graph:
    """
    Converts a table into a graph. Each cell is a node. Edges are created between nodes that are horizontally or
    vertically adjacent.
    :param doc: The document
    :param table_area: The table area
    :return: The graph
    """
    import networkx as nx

    if isinstance(table_area, str):
        table_area = doc.get_area_obj(table_area)
    table_cells = [[doc.get_area_obj(x) for x in y]
                   for y in table_area.data['cells']]

    # create a graph with all cells as nodes  mxn rowxcol
    graph = nx.grid_2d_graph(len(table_cells), len(table_cells[0]))

    pos = {(x, y): (y, -x) for x, y in graph.nodes()}

    # add edges between adjacent cells
    for ir, row in enumerate(table_cells):
        for ic, cell in enumerate(row):
            if cell:
                graph.nodes[(ir, ic)]['content'] = cell.data['content']
                cell = graph.nodes[(ir, ic)]
                # get adjacent cells, search until cell is found
                # if ir > 0:  # top
                #     for i in range(ir-1, -1, -1):
                #         if table_cells[i][ic]:
                #             graph.add_edge(cell,table_cells[i][ic], type='column')
                #             break
                if ir < len(table_cells)-1:  # down
                    for i in range(ir+1, len(table_cells)):
                        if graph.nodes[(ir, ic)]:
                            graph.add_edge(
                                cell, graph.nodes[(ir, ic)], type='column')
                            break

                if ic > 0:  # left
                    for i in range(ic-1, -1, -1):
                        if graph.nodes[(ir, ic)]:
                            graph.add_edge(
                                cell, graph.nodes[(ir, ic)], type='row')
                            break
                # if ic < len(table_cells[ir])-1:  # right
                #     for i in range(ic+1, len(table_cells[ir])):
                #         if table_cells[ir][i]:
                #             graph.add_edge(cell,table_cells[ir][i], type='row')
                #             break

    # debug plot
    nx.draw(graph, pos=pos,
            node_color='lightgreen',
            with_labels=True,
            node_size=600)
    # plt.show()

    return graph


def sort_2D_grid(node_array: List[Area], axis: str = 'x') -> Tuple[List[Area], List[List[Area]]]:
    # ACCREDITATION: Automatic chessboard corner detection method, DOI:10.1049/iet-ipr.2015.0126
    # https://stackoverflow.com/questions/29630052/ordering-coordinates-from-top-left-to-bottom-right

    sorted_nodes = []  # this is the return value
    sorted_grid = []
    available_nodes = list(node_array)  # make copy of input array

    # get minimum node height
    min_height = float('inf')
    for node in available_nodes:
        try:
            min_height = min(min_height, node.boundingBox.height
                             if axis == 'x' else node.boundingBox.width)
        except AttributeError:
            min_height = min(min_height, node.height
                             if axis == 'x' else node.width)

    while len(available_nodes) > 0:
        # find y value of topmost node in availableNodes
        min_y = float('inf')
        for node in available_nodes:
            try:
                min_y = min(min_y, node.boundingBox.y1 if axis ==
                            'x' else node.boundingBox.x1)
            except AttributeError:
                min_y = min(min_y, node.y1 if axis == 'x' else node.x1)

        # find nodes in the top row: assume a node is in the top row when its distance from min_y
        # is less than its height
        top_row = []
        other_rows = []
        for node in available_nodes:
            if axis == 'x':
                try:
                    if abs(min_y - node.boundingBox.y1) <= min_height:
                        top_row.append(node)
                        continue
                except AttributeError:
                    if abs(min_y - node.y1) <= min_height:
                        top_row.append(node)
                        continue
                other_rows.append(node)
            else:
                try:
                    if abs(min_y - node.boundingBox.x1) <= min_height:
                        top_row.append(node)
                        continue
                except AttributeError:
                    if abs(min_y - node.x1) <= min_height:
                        top_row.append(node)
                        continue
                other_rows.append(node)

        # sort the top row by x
        try:
            top_row.sort(key=lambda node: node.boundingBox.x1 if axis ==
                         'x' else node.boundingBox.y1)
        except AttributeError:
            top_row.sort(key=lambda node: node.x1 if axis == 'x' else node.y1)

        # append nodes in the top row to sorted nodes
        sorted_nodes.extend(top_row)
        sorted_grid.append(top_row)

        # update available nodes to exclude handled rows
        available_nodes = other_rows

    return sorted_nodes, sorted_grid
