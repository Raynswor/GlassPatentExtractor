import collections
import itertools
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from kieta_modules import Module
from kieta_data_objs import Area, Document, GroupedAreas, BoundingBox

from kieta_modules.util import UnionFind, find_overlapping_entries, group_horizontally_by_distance, group_vertically_by_alignment, get_overlapping_areas, merge_overlapping_bounding_boxes_in_one_direction, nms_merge, sort_2D_grid


def extend_bounding_boxes_asfaraspossible(matrix: List[List[str]], vertical_lines: List, doc: Document) -> List[List[Area]]:
    # extension has to be in a rectangular shape
    """
    popular patterns
       X    -->  X  X  X
    A  B  C -->  A  B  C

    X  Y  Y -->  X  Y  Y
       A  B -->  X  A  B
    """
    # find these patterns and apply accordingly

    # first pattern
    for (first_row, second_row) in itertools.combinations(range(0, 3), 2):
        try:
            if len(matrix[first_row]) == len(matrix[second_row]) and len(matrix[first_row]) > 1:
                for c in range(1, len(matrix[first_row])-1):
                    if matrix[second_row][c-1] is not None and matrix[first_row][c-1] is None and \
                        matrix[second_row][c] is not None and matrix[first_row][c] is not None and \
                            matrix[second_row][c+1] is not None and matrix[first_row][c+1] is None:
                        # check if there aren't any vertical lines between that
                        x_range = range(int(doc.get_area_obj(matrix[second_row][c-1]).boundingBox.x1), int(
                            doc.get_area_obj(matrix[second_row][c+1]).boundingBox.x2))

                        skip = False
                        for i in vertical_lines:
                            if i.boundingBox.x1 in x_range:
                                skip = True
                                break
                        if not skip:
                            matrix[first_row][c-1] = matrix[first_row][c]
                            matrix[first_row][c+1] = matrix[first_row][c]
        except IndexError:
            pass

    # NOTE: i don't know if that is working or not
    # second pattern  -- only apply to first and second row
    for ic, c in enumerate(matrix[0]):
        try:
            if c is not None and matrix[1][ic] is None:
                matrix[1][ic] = matrix[0][ic]
            elif c is None and matrix[1][ic] is not None:
                matrix[0][ic] = matrix[1][ic]
        except IndexError:
            pass
    return matrix


class HeuristicRecognizer(Module):
    _MODULE_TYPE = 'HeuristicRecognizer'

    def __init__(self, stage: int, parameters: Optional[Dict] = {}, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)

        self.threshold_vertical_translation = parameters.get(
            'threshold_vertical_translation', 5)
        self.apply_to = parameters.get('apply_to', ['TableCell'])
        self.table_grid = parameters.get(
            'table_grid', ['DrawnLine', "LogicalLine", "DrawnRectangle"])
        self.lines = parameters.get('lines', 'Line')
        self.strings = parameters.get('strings', 'String')

    def execute(self, inpt: Document) -> Document:
        for table in list(inpt.get_area_type('Table')):
            page_of_area = inpt.find_page_of_area(table)
            factors = (inpt.pages.byId[page_of_area].factor_width,
                       inpt.pages.byId[page_of_area].factor_height)
            # elements = get_overlapping_areas(table, searchable_areas, ["TableCell", "DrawnLine"])
            elements = get_overlapping_areas(table, inpt.get_area_by(
                lambda _: True, page_of_area), self.apply_to, factors=factors)
            table_grid = get_overlapping_areas(table, inpt.get_area_by(
                lambda _: True, page_of_area), self.table_grid, factors=factors)

            # sort lines into vertical and horizontal lines
            vertical_lines = list()
            horizontal_lines = list()
            try:
                for cat in table_grid.values():
                    for x in cat:
                        if x.boundingBox.y2-x.boundingBox.y1 > x.boundingBox.x2-x.boundingBox.x1:
                            vertical_lines.append(x)
                        else:
                            horizontal_lines.append(x)
                vertical_lines.sort(key=lambda x: x.boundingBox.x1)
                horizontal_lines.sort(key=lambda x: x.boundingBox.y1)
            except KeyError:
                pass

            threshold_vertical_translation = self.threshold_vertical_translation
            compare_groups: List[List[GroupedAreas]] = list()
            probable_col_numbers: List[int] = list()

            # TODO: This approach does not really work well with sparse tables

            for _ in range(3):
                groups: List[GroupedAreas] = []
                # Arrange to rows
                for k in elements:
                    random.shuffle(elements[k])

                flattened = list(
                    itertools.chain.from_iterable(elements.values()))

                for obj in flattened:
                    img_space_bb = obj.boundingBox.get_in_img_space(factors[0], factors[1])

                    done = False
                    # Iterate over other objects
                    for other_group in groups:
                        for other in other_group.areas:
                            other_img_space_bb = other.boundingBox.get_in_img_space(factors[0], factors[1])

                            # check if top or bot align
                            if (img_space_bb.y1 - threshold_vertical_translation <= other_img_space_bb.y1 <= img_space_bb.y1 + threshold_vertical_translation or
                                    img_space_bb.y2 - threshold_vertical_translation <= other_img_space_bb.y2 <= img_space_bb.y2 + threshold_vertical_translation):
                                # check that no horizontal line is crossed
                                s1 = range(int(min(img_space_bb.y1, other_img_space_bb.y1)), int(
                                    max(img_space_bb.y2, other_img_space_bb.y2)))

                                skip = False
                                for i in horizontal_lines:
                                    if i.boundingBox.y1 in s1:
                                        skip = True
                                        break
                                if skip:  # todo: break or continue??
                                    break

                                # check that no vertical overlap exists to another member of the group
                                for other2 in other_group.areas:
                                    other2_img_space_bb = other2.boundingBox.get_in_img_space(factors[0], factors[1])
                                    if other2_img_space_bb.overlap_horizontally(img_space_bb):
                                        skip = True
                                        break
                                if skip:
                                    break
                                other_group.areas.append(obj)
                                done = True
                                break
                        # if done:
                        #     break
                    if not done:
                        groups.append(GroupedAreas([obj]))

                for g in groups:
                    g.areas.sort(key=lambda x: x.boundingBox.x1)
                    # debug
                    inpt.add_area(page_of_area, "Row", g.get_boundingBox(), convert_to_xml=True)
                groups.sort(key=lambda x: x.get_boundingBox().y1)

                for g in groups:
                    print([x.data['content'] if x else "----" for x in g.areas])

                # groups = sort_2D_grid(groups)
                compare_groups.append(groups)

                # self.debug_msg(collections.Counter([len(x.areas) for x in groups]))
                try:
                    probable_col_numbers.append(max(collections.Counter(
                        [len(x.areas) for x in groups]).most_common(3), key=lambda x: x[0])[0])
                except ValueError:
                    probable_col_numbers.append(0)

            # self.debug_msg("cols", collections.Counter(probable_col_numbers))
            probable_col_number = collections.Counter(
                probable_col_numbers).most_common(1)[0][0]
            self.debug_msg(f"probable col number {probable_col_number}")

            distances = list()
            for ix, c in enumerate(compare_groups):
                # calculate distance to most likely column number
                score = 0
                # self.debug_msg(f"GROUP {ix}")
                for cc in c:
                    score += abs(probable_col_number - len(cc.areas))
                    # self.debug_msg([ccc.data['content'] for ccc in cc.areas])
                    if probable_col_number - len(cc.areas) > 0:  # areas missing
                        score += 0.5*(probable_col_number - len(cc.areas))
                    else:
                        score += abs(probable_col_number - len(cc.areas))
                    # print([ccc.data['content'] for ccc in cc.areas])
                distances.append(score)
            groups = compare_groups[np.argmin(distances)]

            self.debug_msg(f"best configuration is no {np.argmin(distances)}:")
            # print config
            # self.debug_msg(pd.DataFrame([[rr.data['content'] if rr else "----" for rr in r.areas] for r in groups]))

            # [rows][columns]
            matrix: List[List] = [[None for _ in range(
                probable_col_number)] for _ in range(len(groups))]
            # pre-set all rows with probable_col_number columns
            tbd = list()
            for ix, s in enumerate(groups):
                if len(s.areas) == probable_col_number:
                    matrix[ix] = [x.oid for x in s.areas]
                    matrix_map = {x.oid: (ix, ixx)
                                  for ixx, x in enumerate(s.areas)}
                else:
                    tbd.append(ix)

            # TODO: big empty spaces are a problem
            # TODO: sometimes cells are overwritten (probably when something doesn't fit )
            for rx in tbd:
                x = groups[rx]
                # if less columns, find best match
                if len(x.areas) < probable_col_number:
                    # for this row rx find next not-none cell in matrix
                    above, below = list(), list()
                    for cx in range(probable_col_number):
                        above_rx, below_rx = rx-1, rx+1
                        try:
                            while matrix[above_rx][cx] is None:
                                above_rx -= 1
                            above.append(matrix[above_rx][cx])
                        except IndexError:
                            above.append(None)
                        try:
                            if matrix[below_rx][cx] is None:
                                below_rx += 1
                            below.append(matrix[below_rx][cx])
                        except IndexError:
                            below.append(None)

                    # compare with above and below
                    for a in x.areas:
                        for i, (ab, be) in enumerate(zip(above, below)):
                            if ab is not None and a.boundingBox.overlap_horizontally(inpt.areas.byId[ab].boundingBox):
                                matrix[rx][i] = a.oid
                                matrix_map[a.oid] = (rx, i)
                                break
                            elif be is not None and a.boundingBox.overlap_horizontally(inpt.areas.byId[be].boundingBox):
                                matrix[rx][i] = a.oid
                                matrix_map[a.oid] = (rx, i)
                                break

                # if more columns, find best match
                # decide for excess cells if they should be merged or pushed to next row
                else:
                    pass
                    # self.debug_msg("excess columns")
                    # self.debug_msg([ccc.data['content'] for ccc in x])
                    # to_merge: Dict[str, Area] = dict()
                    # for cx, a in enumerate(seed_row.areas):
                    #     if matrix[rx][cx] is None:
                    #         for a2 in x.areas:
                    #             if a.boundingBox.overlap_horizontally(a2.boundingBox):
                    #                 matrix[rx][cx] = a2.oid
                    #                 matrix_map[a2.oid] = (rx, cx)
                    #                 break
                    #     # handle excess cells --> current approach: just merge
                    #     else:
                    #         to_merge[matrix[rx][cx]] = a

                    # merge excess cells
                    # for oid, a in to_merge.items():
                    #     # self.debug_msg(f"merging {oid} with {a.oid}, with content {inpt.areas.byId[oid].data['content']} and {a.data['content']}")
                    #     inpt.areas.byId[oid].merge(a)
                    #     inpt.delete_area(a.oid)

            # debug output result
            self.debug_msg("resulting table")
            self.debug_msg(pd.DataFrame(
                [[inpt.areas.byId[rr].data['content'] if rr else "----" for rr in r] for r in matrix]))

            table.data["cells"] = matrix

        return inpt


class MarkovChainRecognizer(Module):
    _MODULE_TYPE = 'MarkovChainRecognizer'

    def __init__(self, stage: int, parameters: Optional[Dict] = {}, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)

        self.apply_to = parameters.get('apply_to', ['TableCell'])
        self.table_grid = parameters.get(
            'table_grid', ['DrawnLine', "LogicalLine", "DrawnRectangle"])
        self.max_iterations = parameters.get('max_iterations', 10)
        self.threshold_vertical_translation = parameters.get(
            'threshold_vertical_translation', 0)
        self.convergence_threshold = parameters.get(
            'convergence_threshold', 0.1)

        self.lines = parameters.get('lines', 'Line')
        self.strings = parameters.get('strings', 'String')

    def is_converged(self, previous_groups, current_groups, threshold=0.1):
        """
        Checks if the grouping has converged based on a defined threshold.

        :param previous_groups: The groups from the previous iteration.
        :param current_groups: The groups from the current iteration.
        :param threshold: A threshold to determine the similarity required for convergence.
        :return: True if the system is considered converged, False otherwise.
        """

        if len(previous_groups) != len(current_groups):
            # If the number of groups changed, it's not converged
            return False

        total_differences = 0
        max_group_size = max(len(g) for g in current_groups)

        for prev_group, curr_group in zip(previous_groups, current_groups):
            # Count how many members of the group are different
            differences = len(set(prev_group) ^ set(curr_group))
            normalized_diff = differences / max_group_size
            total_differences += normalized_diff

        average_difference = total_differences / len(current_groups)

        return average_difference <= threshold

    def markov_chain_grouping_rows(self,
                                   elements,
                                   lines,
                                   max_iterations=100,
                                   threshold_vertical_translation=5,
                                   convergence_threshold=1):
        
        probable_col_numbers: List[int] = list()
        previous_groups: List[GroupedAreas] = None
        current_groups: List[GroupedAreas] = None
        iterations = 0
        line_bbs = np.array([l.boundingBox.tuple() for l in lines])

        while iterations < max_iterations:
            random.shuffle(elements)

            _, groups = merge_overlapping_bounding_boxes_in_one_direction(
                [obj.boundingBox for obj in elements], 10000, -threshold_vertical_translation, line_bbs)

            # Check for convergence
            if previous_groups is not None:
                if self.is_converged(previous_groups, groups, convergence_threshold):
                    # self.debug_msg(f"Converged after {iterations} iterations")
                    break

            previous_groups = current_groups
            current_groups = groups
            iterations += 1

            try:
                # self.debug_msg(f"most common number of entries per row: {collections.Counter([len(x) for x in groups]).most_common(3)}")
                probable_col_numbers.append(max(collections.Counter(
                    [len(x) for x in groups]).most_common(3), key=lambda x: x[0])[0])
            except ValueError:
                probable_col_numbers.append(0)
        
        # current group to grouped areas
        current_groups =  [GroupedAreas(sorted([elements[i] for i in v], key=lambda x: x.boundingBox.x1)) for v in current_groups]
        current_groups.sort(key=lambda x: x.get_boundingBox().y1)

        probable_col_number = collections.Counter(
            probable_col_numbers).most_common(1)[0][0]
        self.debug_msg(f"markov needed {iterations} iterations")
        self.debug_msg(f"mine - from {len(elements)} to {len(current_groups)} groups")
        self.debug_msg(f"probable col number {probable_col_number}")

        return current_groups, probable_col_number

    def execute(self, inpt: Document) -> Document:
        l_tables = list(inpt.get_area_type('Table'))
        for table in l_tables:
            searchable_areas = list()
            factors = (1, 1)
            for k in inpt.pages.allIds:
                if table.oid in inpt.references.byId[k]:
                    searchable_areas = [inpt.areas.byId[kk]
                                        for kk in inpt.references.byId[k]]
                    factors = (
                        inpt.pages.byId[k].factor_width, inpt.pages.byId[k].factor_height)
                    # table.boundingBox = table.boundingBox.get_in_img_space(inpt.pages.byId[k].factor_width, inpt.pages.byId[k].factor_height)
                    break
            elements = get_overlapping_areas(
                table, searchable_areas, self.apply_to, factors=factors)
            
            # TODO: implement that the grid is also used in some way
            # elements = get_overlapping_areas(table, searchable_areas, [self.strings, "DrawnLine"])

            cells = elements.get("TableCell", [])

            # sort lines into vertical and horizontal lines
            table_grid = get_overlapping_areas(
                table, searchable_areas, self.table_grid, factors=factors)

            # sort lines into vertical and horizontal lines
            vertical_lines = list()
            horizontal_lines = list()
            try:
                for cat in table_grid.values():
                    for x in cat:
                        if x.boundingBox.y2-x.boundingBox.y1 > x.boundingBox.x2-x.boundingBox.x1:
                            vertical_lines.append(x)
                        else:
                            horizontal_lines.append(x)
                vertical_lines.sort(key=lambda x: x.boundingBox.x1)
                horizontal_lines.sort(key=lambda x: x.boundingBox.y1)
            except KeyError:
                pass

            groups, probable_col_number = self.markov_chain_grouping_rows(cells,
                                                                          horizontal_lines,
                                                                          convergence_threshold=self.convergence_threshold,
                                                                          max_iterations=self.max_iterations,
                                                                          threshold_vertical_translation=self.threshold_vertical_translation)

            # add rows
            for g in groups:
                print(g.get_boundingBox(), [x.data['content'] for x in g.areas], [x.boundingBox.tuple() for x in g.areas])
                inpt.add_area(k, "Row", g.get_boundingBox(), convert_to_xml=not g.areas[0].boundingBox.img_sp)
                g.areas.sort(key=lambda x: x.boundingBox.x1)  # IMPORTANT

            # distances = list()
            # for ix, c in enumerate(compare_groups):
            #     # calculate distance to most likely column number
            #     score = 0
            #     # self.debug_msg(f"GROUP {ix}")
            #     for cc in c:
            #         score += abs(probable_col_number - len(cc.areas))
            #         # self.debug_msg([ccc.data['content'] for ccc in cc.areas])
            #         if probable_col_number - len(cc.areas) > 0:  # areas missing
            #             score += 0.5*(probable_col_number - len(cc.areas))
            #         else:
            #             score += abs(probable_col_number - len(cc.areas))
            #         # print([ccc.data['content'] for ccc in cc.areas])
            #     distances.append(score)
            # groups = compare_groups[np.argmin(distances)]

            # self.debug_msg(f"best configuration is no {np.argmin(distances)}:")
            # print config
            # self.debug_msg(pd.DataFrame([[rr.data['content'] if rr else "----" for rr in r.areas] for r in groups]))

            # [rows][columns]
            matrix: List[List] = [[None for _ in range(
                probable_col_number)] for _ in range(len(groups))]
            # pre-set all rows with probable_col_number columns
            tbd = list()
            for ix, s in enumerate(groups):
                if len(s.areas) == probable_col_number:
                    matrix[ix] = [x.oid for x in s.areas]
                    matrix_map = {x.oid: (ix, ixx)
                                  for ixx, x in enumerate(s.areas)}
                else:
                    tbd.append(ix)

            # TODO: big empty spaces are a problem
            # TODO: sometimes cells are overwritten (probably when something doesn't fit )
            for rx in tbd:
                x = groups[rx]
                # if less columns, find best match
                if len(x.areas) < probable_col_number:
                    # for this row rx find next not-none cell in matrix
                    above, below = list(), list()
                    for cx in range(probable_col_number):
                        above_rx, below_rx = rx-1, rx+1
                        try:
                            while matrix[above_rx][cx] is None:
                                above_rx -= 1
                            above.append(matrix[above_rx][cx])
                        except IndexError:
                            above.append(None)
                        try:
                            if matrix[below_rx][cx] is None:
                                below_rx += 1
                            below.append(matrix[below_rx][cx])
                        except IndexError:
                            below.append(None)

                    # TODO: get all below and above and then use the one where more overlap is

                    # compare with above and below
                    for a in x.areas:
                        for i, (ab, be) in enumerate(zip(above, below)):
                            if ab is not None and a.boundingBox.overlap_horizontally(inpt.areas.byId[ab].boundingBox) and matrix[rx][i] is None:
                                matrix[rx][i] = a.oid
                                matrix_map[a.oid] = (rx, i)
                                break
                            elif be is not None and a.boundingBox.overlap_horizontally(inpt.areas.byId[be].boundingBox) and matrix[rx][i] is None:
                                matrix[rx][i] = a.oid
                                matrix_map[a.oid] = (rx, i)
                                break
                # if more columns, find best match
                # decide for excess cells if they should be merged or pushed to next row
                else:
                    self.debug_msg("has more columns than probable_col_number")
                    pass

            # debug output result
            # self.debug_msg("resulting table")
            # try:
            #     self.debug_msg(pd.DataFrame([[[inpt.areas.byId[ref].data['content']
            #                    for ref in inpt.references.byId[rr]] if rr else "----" for rr in r] for r in matrix]))
            # except:
            #     self.debug_msg(pd.DataFrame(
            #         [[inpt.areas.byId[rr].data['content'] if rr else "----" for rr in r] for r in matrix]))

            table.data["cells"] = matrix

            # for r in matrix:
            #     print(r)

            # NOTE: does it make a difference?
            matrix = extend_bounding_boxes_asfaraspossible(
                matrix, vertical_lines, doc=inpt)

            # for r in matrix:
            #     print(r)

        return inpt
