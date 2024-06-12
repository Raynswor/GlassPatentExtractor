import itertools
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from .. import Module
from ..util import UnionFind, get_overlapping_areas, sort_into_two_lists

from kieta_data_objs import Document, BoundingBox, Area

from typing import Dict, Optional, List

import numpy as np
import cv2
import doxapy

from PIL import Image


class LineDetectorCanny(Module):
    _MODULE_TYPE = "LineDetectorCanny"

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = None) -> None:
        super().__init__(stage, parameters, debug_mode)

        self.max_line_gap = parameters.get("max_line_gap", 10)
        self.min_line_length = parameters.get("min_line_length", 200)
        self.minimum_number_of_votes = parameters.get("minimum_number_of_votes", 10)

        from sklearn.cluster import DBSCAN
        from sklearn.linear_model import LinearRegression
        self.linreg = LinearRegression()
        self.dbscan = DBSCAN(eps=25, min_samples=2)

    def execute(self, doc: Document) -> Document:
        for p_ix in doc.pages.allIds:
            for line in self.detect_lines(
                doc.get_img_page(p_ix, False), self.max_line_gap):
                doc.add_area(p_ix, "DrawnLine", boundingBox=line)
        return doc
    
    # @profile
    def detect_lines(self, image, max_possible_gap: int = 10) -> List[Area]:
        image = np.array(image)

        dilatation_dst = np.empty((image.shape[0], image.shape[1]), dtype=np.uint8)

        # thresholding
        # sauvola = doxapy.Binarization(doxapy.Binarization.Algorithms.SAUVOLA)
        sauvola = doxapy.Binarization(doxapy.Binarization.Algorithms.SAUVOLA)
        sauvola.initialize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        sauvola.to_binary(dilatation_dst, parameters={"window": 15, "k": 0.2, "R": 68})


        # dilatation 
        # inverse picture
        dilatation_dst = cv2.bitwise_not(dilatation_dst)
        dilatation_size = 3
        element = cv2.getStructuringElement( cv2.MORPH_CROSS, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
        dilatation_dst = cv2.dilate(dilatation_dst, element)
        dilatation_dst = cv2.bitwise_not(dilatation_dst)
        # Image.fromarray(dilatation_dst).save(f"img.png")


        ### Canny
        low_threshold = 100
        high_threshold = 110
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid

        # Run Hough on edge detected image
        lines = cv2.HoughLinesP(cv2.Canny(dilatation_dst, low_threshold, high_threshold), 
                                rho, 
                                theta, 
                                self.minimum_number_of_votes,
                                np.array([]), 
                                minLineLength=self.min_line_length, 
                                maxLineGap=max_possible_gap)

        res = list()
        if lines is None:
            return res

        for line in lines:
            x1, y1, x2, y2 = line[0]
            res.append(BoundingBox(int(x1), int(y1), int(x2), int(y2), img_sp=True))
        #     cv2.line(dilatation_dst, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Image.fromarray(dilatation_dst).save(f"lines.png")

        # sort in vertical and horizontal lines
        horizontal, vertical = sort_into_two_lists(res, lambda x: x.is_horizontal())

        # logger.debug(f"found {len(res)} lines, {len(horizontal)} horizontal and {len(vertical)} vertical")
        # extend lines as long as there are black pixels in the direction
        debug_extension_log = list()
        for line in horizontal:
            # extend left
            x = line.x1
            y = line.y1
            while x > 0 and (dilatation_dst[y-1:y+2, x-max_possible_gap:x] <= 200).any():
                x -= 1
            if line.x1 != x:
                debug_extension_log.append(f"extended line {line} to {x},{y}")
                line.x1 = x
            # extend right
            x = line.x2
            y = line.y1
            while x < dilatation_dst.shape[1] and (dilatation_dst[y-1:y+2, x:x+max_possible_gap] <= 200).any():
                x += 1
            if line.x2 != x:
                debug_extension_log.append(f"extended line {line} to {x},{y}")
                line.x2 = x
            # check if x2 > x1 and swap if not
            # cv2.line(image, (line.x1, line.y1), (line.x2, line.y2), (0, 125, 125), 2)
        for line in vertical:
            # extend up
            x = line.x1
            y = line.y1
            while y > 0 and (dilatation_dst[y-max_possible_gap:y, x-1:x+2] <= 200).any():
                y -= 1
            if line.y1 != y:
                debug_extension_log.append(f"extended line {line} to {x},{y}")
                line.y1 = y
            # extend down
            x = line.x1
            y = line.y2
            while y < dilatation_dst.shape[0] and (dilatation_dst[y:y+max_possible_gap, x-1:x+2] <= 200).any():
                y += 1
            if line.y2 != y:
                debug_extension_log.append(f"extended line {line} to {x},{y}")
                line.y2 = y
        #     cv2.line(image, (line.x1, line.y1), (line.x2, line.y2), (0, 125, 125), 3)

        # Image.fromarray(image).save(f"lines_edges.png")


        result = list()
        try:
            verts = self.optimize_lines_v2(horizontal, True)
            horts = self.optimize_lines_v2(vertical, False)
            result = verts + horts
        except Exception as e:
            print(e)
            pass

        return result


    def optimize_lines_v2(self, lines: List[BoundingBox], direction: bool) -> List[BoundingBox]:
        """
        Optimizes lines by using DBSCAN and linear regression
        """
        # cluster lines
        X = list()

        if not lines:
            return list()

        for l in lines:
            for x in range(l.x1, l.x2+1):
                for y in range(l.y1, l.y2+1):
                    X.append((x, y))
        X = np.array(X)

        clustering = self.dbscan.fit(X)

        labels = clustering.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        # plot clusters
        # unique_labels = set(labels)
        # colors = [plt.cm.Spectral(each)
        #           for each in np.linspace(0, 1, len(unique_labels))]
        # for k, col in zip(unique_labels, colors):
        #     if k == -1:
        #         col = [0, 0, 0, 1]

        #     class_member_mask = labels == k
        #     xy = X[class_member_mask]
        #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(
        #         col), markeredgecolor='k', markersize=6)

        # fit lines
        res = list()
        for k in range(n_clusters_):
            xy = X[labels == k]
            if len(xy) < 2:
                continue
            
            if direction:
                reg = self.linreg.fit(xy[:, 0].reshape(-1, 1), xy[:, 1])
                x1 = min(xy[:, 0])
                x2 = max(xy[:, 0])
                y1 = reg.predict([[x1]])[0]
                y2 = reg.predict([[x2]])[0]
                # ensure at least 1 px width

            else:
                reg = self.linreg.fit(xy[:, 1].reshape(-1, 1), xy[:, 0])
                y1 = min(xy[:, 1])
                y2 = max(xy[:, 1])
                x1 = reg.predict([[y1]])[0]
                x2 = reg.predict([[y2]])[0]
                # ensure at least 1 px width
            if x1 == x2:
                x1 -= 1
                x2 += 1
            if y1 == y2:
                y1 -= 1
                y2 += 1

            # plot regression line
            # plt.plot([x1, x2], [y1, y2], color='pink', linewidth=2)

            res.append(BoundingBox(round(min(x1, x2)), round(min(y1, y2)), round(max(x1, x2)), round(max(y1, y2)), img_sp=True))
        
        # plt.show()
        self.debug_msg(f"optimized {len(lines)} to {len(res)} lines")

        return res
    
def plot_clustering(X, labels):
    from matplotlib import pyplot as plt
    # plot clusters
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
                for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = labels == k
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(
            col), markeredgecolor='k', markersize=6)
    plt.show()


class LogicalLineGuesser(Module):
    _MODULE_TYPE = 'LogicalLineGuesser'

    def __init__(self, stage: int, parameters: Optional[Dict] = {}, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)

        self.apply_to = parameters.get('apply_to', ['Table'])
        self.subelement = parameters.get('subelement', 'TableCell')

        self.gap_between_rows = parameters.get('gap_between_rows', 3)
        self.gap_between_columns = parameters.get('gap_between_columns', 3)

        # how many parts the table should be divided into
        self.h_split = parameters.get('h_split', 1)
        self.v_split = parameters.get('v_split', 2)

        self.merge_radius_vertical = parameters.get('merge_radius_vertical', 20)
        self.merge_radius_horizontal = parameters.get('merge_radius_horizontal', 20)

    def execute(self, inpt: Document) -> Document:
        for table in list(inpt.get_area_type('Table')):
            page_of_area = inpt.find_page_of_area(table)
            factors = (inpt.pages.byId[page_of_area].factor_width,
                       inpt.pages.byId[page_of_area].factor_height)
            # elements = get_overlapping_areas(table, searchable_areas, ["TableCell", "DrawnLine"])
            elements = get_overlapping_areas(table, inpt.get_area_by(
                lambda _: True, page_of_area), [self.subelement, "DrawnLine"], factors=factors)
            drawn_lines = elements.get("DrawnLine", [])
            elements = elements.get(self.subelement, []) 

            if not elements or len(elements) < 4:
                self.warning_msg("Table has less than 4 elements, skipping")
                continue

            # homogenize to img space
            bbs = [l.boundingBox.get_in_img_space(*factors) for l in elements]

            # find horizontal and vertical middle
            h_mid, v_mid = list(), list()
            for l in bbs:
                h_mid.append((l.x1 + l.x2) / 2)
                v_mid.append((l.y1 + l.y2) / 2)
            
            # divide into h_split and v_split parts
            h_mids = [range(int(min(x)), int(max(x))) for x in np.array_split(np.sort(h_mid), self.h_split) if len(x) >= 1]
            v_mids = [range(int(min(x)), int(max(x))) for x in np.array_split(np.sort(v_mid), self.v_split) if len(x) >= 1]
            
            vertical_splits, horizontal_splits = [set() for _ in range(self.v_split)], [set() for _ in range(self.h_split)]

            for l in bbs:
                for x in range(int(l.x1), int(l.x2)+1):
                    for y in range(int(l.y1), int(l.y2)+1):
                        for i, h in enumerate(h_mids):
                            if x in h:
                                horizontal_splits[i].add((x, y))
                        for i, v in enumerate(v_mids):
                            if y in v:
                                vertical_splits[i].add((x, y))

            # # plot
            # import matplotlib.pyplot as plt
            # plt.figure()
            # for y in range(self.v_split):
            #     plt.subplot(2, max(self.v_split, self.h_split), y+1)
            #     plt.title(f"Vertical split {y}")
            #     plt.gca().invert_yaxis()
            #     plt.scatter([i[0] for i in vertical_splits[y]], [i[1] for i in vertical_splits[y]])
            #     for middle in self.calculate_gaps([x[0] for x in vertical_splits[y]], self.vertical_gap):
            #         # plt line
            #         print(middle)
            #         plt.axvline(middle, color='r')

            # for x in range(self.h_split):
            #     plt.subplot(2, max(self.v_split, self.h_split), self.v_split + x + 1)
            #     # title
            #     plt.title(f"Horizontal split {x}")
            #     plt.scatter([i[0] for i in horizontal_splits[x]], [i[1] for i in horizontal_splits[x]], c='r')
            #     for middle in self.calculate_gaps([x[1] for x in horizontal_splits[x]], self.horizontal_gap):
            #         # plt line
            #         plt.axhline(middle, color='r')
            # plt.show()

            table_img_bb = table.boundingBox.get_in_img_space(*factors)

            vertical_lines = self.process_lines(vertical_splits, v_mids, self.gap_between_columns, table_img_bb, 0, self.merge_radius_vertical)
            for l, _ in vertical_lines:
                inpt.add_area(page_of_area, "LogicalLine", l)

            horizontal_lines = self.process_lines(horizontal_splits, h_mids, self.gap_between_rows, table_img_bb, 1, self.merge_radius_horizontal)
            for l, _ in horizontal_lines:
                inpt.add_area(page_of_area, "LogicalLine", l)

        return inpt
    
    def process_lines(self, splits, mids, gap, bb, direction, distance):
        lines = list()
        for ix, v in enumerate(splits):
            try:
                start, end = mids[ix][0], mids[ix][-1]
            except IndexError:
                continue
            for middle in self.calculate_gaps([x[direction] for x in v], gap):
                if direction == 0:  # vertical
                    lines.append((BoundingBox(middle, max(start, bb.y1), middle+1, end, img_sp=True), ix))
                else:  # horizontal
                    lines.append((BoundingBox(max(start, bb.x1), middle, end, middle+1, img_sp=True), ix))
        return self.merge_lines(lines, distance)
    
    def merge_lines(self, lines, distance) -> List[BoundingBox]:
        before = len(lines)
        self.debug_msg(f"found {before} horizontal lines")
        while True:         
            uf = UnionFind.from_count(len(lines))
            for i in range(len(lines)):
                for j in range(i+1, len(lines)):
                    if lines[i][1]!=lines[j][1] and lines[i][0].distance(lines[j][0]) < distance:
                        uf.union2(i, j)
            new_lines = list()
            for sets in uf.get_sets().values():
                if len(sets) == 1:
                    new_lines.append(lines[sets[0]])
                else:
                    x1 = min([lines[l][0].x1 for l in sets])
                    x2 = max([lines[l][0].x2 for l in sets])
                    y1 = min([lines[l][0].y1 for l in sets])
                    y2 = max([lines[l][0].y2 for l in sets])
                    new_lines.append((BoundingBox(x1, y1, x2, y2, img_sp=True), lines[sets[0]][1]))
            if len(new_lines) == before:
                break
            before = len(new_lines)
            lines = new_lines
        self.debug_msg(f"optimized to {len(lines)} horizontal lines")
        return lines

    def calculate_gaps(self, X, gap_threshold):
        ranges = list(sorted(set(X)))
        for i in range(len(ranges)-1):
            pass
            if ranges[i+1] - ranges[i] > gap_threshold:
                yield (ranges[i] + ranges[i+1]) * 0.5
