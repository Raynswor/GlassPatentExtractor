from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from kieta_modules.util import UnionFind, merge_overlapping_bounding_boxes_in_one_direction
from PIL import ImageDraw, Image, ImageOps
import copy
import time
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
from kieta_modules.base import Module
from kieta_modules import util

from kieta_data_objs import Document, BoundingBox, Area

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


logging.getLogger("segmentation").setLevel(logging.CRITICAL)
logging.getLogger("segmentation-pytorch").setLevel(logging.CRITICAL)
logging.getLogger("pixel_classifier_torch").setLevel(logging.CRITICAL)


class BaselineDetector(Module):
    _MODULE_TYPE = 'BaselineDetector'

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = False) -> None:

        from segmentation.model_builder import ModelBuilderLoad
        from segmentation.network import EnsemblePredictor
        from segmentation.network_postprocessor import NetworkBaselinePostProcessor
        from segmentation.preprocessing.source_image import SourceImage

        super().__init__(stage, parameters, debug_mode)
        model_path = parameters.get('model_path', [])
        if model_path is None:
            raise ValueError('No model path provided for baseline detector')

        self.whitespace_size = parameters.get('whitespace_size', 15)
        self.max_width = parameters.get('max_width', 1000)
        self.max_height = parameters.get('max_height', 25)

        base_model_files = [ModelBuilderLoad.from_disk(model_weights=i, device=parameters.get('device', 'cuda'))
                            for i in model_path]
        base_models = [i.get_model() for i in base_model_files]
        base_configs = [i.get_model_configuration() for i in base_model_files]
        preprocessing_settings = [i.get_model_configuration(
        ).preprocessing_settings for i in base_model_files]

        predictor = EnsemblePredictor(base_models, preprocessing_settings)
        config = base_configs[0]

        self.nbaselinepred = NetworkBaselinePostProcessor(
            predictor, config.color_map)

    def execute(self, inpt: Document) -> Document:
        for page in self.get_progress_bar(inpt.pages.allIds, unit="pages"):
            img = inpt.get_img_page(page, False)
            bls, blines = self.do_single_page(img)

            for bb, bl in zip(bls, blines):
                inpt.add_area(page, "OCRLine", bb, data={"baseline": bl})
        return inpt

    def do_single_page(self, img) -> List[BoundingBox]:
        np_img = np.array(img)
        # black to white, white to black
        np_img = 255 - np_img

        # list of polylines
        # where each polyline is a list of xy-tuples
        result: List[List[Tuple[int, int]]
                     ] = self.nbaselinepred.predict_image(SourceImage(img))

        baselines = [self.straigthen(bl) for bl in result.base_lines]
        baselines = [self.optimize_baseline(bl, np_img) for bl in baselines]

        bases = copy.copy(baselines)

        top_baselines = [self.get_top(bl, np_img, 15) for bl in baselines]
        imprv_top = [self.optimize_top_baseline(
            bl, np_img) for bl in top_baselines]

        bls = list()
        for (bot, top) in zip(baselines, imprv_top):
            bls.append(BoundingBox(
                top[0][0], top[0][1], bot[-1][0], bot[-1][1], img_sp=True))

        # merge baselines that are close together
        bls, bases = self.merge_neighboring_baselines(
            bls, bases, img.size, self.whitespace_size)
        return self.optimize_boundingboxes(bls, np.array(ImageOps.invert(img.convert("L"))).astype(np.int16)), bases

    def optimize_boundingboxes(self, bbs: List[BoundingBox], img) -> List[BoundingBox]:
        # move edges of bounding so that they are on the edge of the text
        # aka each edge of the bounding sits exclusively on white pixels

        # it is necessary to do that round after round because it is possible that a move in one direction
        # opens up a new space in another direction

        # TODO: consider max width and height
        #   -> max height: do not stop at that , but go row/column with least black pixels
        #   -> additionally: stop at other baselines DONE
        for bb in bbs:
            # mask all bounding boxes
            img[bb.y1:bb.y2, bb.x1:bb.x2] = -1000000

        for bb in bbs:
            # move left edge
            change = True
            while change:
                change = False
                if np.sum(img[bb.y1:bb.y2, bb.x1-1]) > 0:
                    bb.x1 -= 1
                    change = True
                # move right edge
                if np.sum(img[bb.y1:bb.y2, bb.x2+1]) > 0:
                    bb.x2 += 1
                    change = True
                # move top edge
                if np.sum(img[bb.y1-1, bb.x1:bb.x2]) > 0:
                    bb.y1 -= 1
                    change = True
                # move bottom edge
                if np.sum(img[bb.y2+1, bb.x1:bb.x2]) > 0:
                    bb.y2 += 1
                    change = True
        return bbs

    def merge_neighboring_baselines(self, bbs: List[BoundingBox], bases, dims, whitespace) -> List[BoundingBox]:
        # iterate through baselines
        # starting from middle of each baseline go right. Merge with the first baseline that is close enough
        mtrx = np.zeros((dims[1], dims[0]))
        # reshape to 2d Matrix
        mtrx -= 1

        # create 2D matrix of baseline indices
        for ix, bb in enumerate(bbs):
            mtrx[bb.y1:bb.y2, bb.x1:bb.x2] = ix

        old_size = -1
        # find closest column containing non-negative values
        while old_size != len(bbs):
            # THIS IS CORRECT  Y HEIGHT, X WIDTH
            old_size = len(bbs)
            for ix, bb in enumerate(bbs):
                idx = None
                # go through columns of this array
                for x in mtrx[bb.y1:bb.y2, bb.x2:bb.x2+whitespace].T:
                    if np.any(x >= 0):
                        # get this value
                        idx = int(np.min(x[x >= 0]))
                        break
                # if no column found, continue
                if idx is None:
                    continue
                else:
                    bbs[ix].x1 = min(bbs[idx].x1, bbs[ix].x1)
                    bbs[ix].x2 = max(bbs[idx].x2, bbs[ix].x2)
                    bbs[ix].y1 = min(bbs[idx].y1, bbs[ix].y1)
                    bbs[ix].y2 = max(bbs[idx].y2, bbs[ix].y2)
                    del bbs[idx]
                    del bases[idx]
                    break
                # Image.fromarray(mtrx[bb.y1:bb.y2, bb.x2:bb.x2+whitespace]).save("/tmp/OUTPUT/test1.png")
            # create 2D matrix of baseline indices
            for ix, bb in enumerate(bbs):
                mtrx[bb.y1:bb.y2, bb.x1:bb.x2] = ix
        return bbs, bases

    def straigthen(self, baseline: List[Tuple[int, int]]):
        min_x = min(baseline, key=lambda x: x[0])[0]
        max_x = max(baseline, key=lambda x: x[0])[0]
        min_y = min(baseline, key=lambda x: x[1])[1]
        max_y = max(baseline, key=lambda x: x[1])[1]

        middle_y = round((min_y + max_y) / 2)

        return [(ix, middle_y) for ix in range(min_x, max_x + 1)]

    def optimize_baseline(self, baseline: List[Tuple[int, int]], img: np.ndarray, direction: int = 1):
        # 1 is down
        # move baseline down that so the number of black pixels directly above it are maximized
        # this is a greedy algorithm, but it should work well enough
        min_x = baseline[0][0]
        max_x = baseline[-1][0]
        current_y = baseline[0][1]

        # get number of black pixels above and on the baseline
        above = np.sum(img[current_y - direction, min_x:max_x + 1])
        on_it = np.sum(img[current_y, min_x:max_x + 1])

        while on_it > above:
            current_y = current_y + direction
            above = np.sum(img[current_y - direction, min_x:max_x + 1])
            on_it = np.sum(img[current_y, min_x:max_x + 1])

        return [(ix, current_y) for ix in range(min_x, max_x + 1)]

    def get_top(self, baseline: List[Tuple[int, int]], img: np.ndarray, font_size: int):
        # assume a certain font size
        # then, we can calculate the height of the baseline

        # get the height of the baseline
        top = baseline[0][1] - font_size

        return [(ix, top) for ix, _ in baseline]

    def optimize_top_baseline(self,
                              baseline: List[Tuple[int, int]],
                              img: np.ndarray,
                              tolerance: int = 2,
                              max_expand: int = 15):
        # 1 is down
        # move baseline down that so the number of black pixels directly above it are maximized
        # this is a greedy algorithm, but it should work well enough

        break_off = 0
        top = baseline[0][1]

        while np.sum(img[top, [i[0] for i in baseline]]) > tolerance and break_off < max_expand:
            top = top - 1
            break_off += 1

        return [(ix, top) for ix, _ in baseline]


class ConnectedComponentOCRModule(Module):
    _MODULE_TYPE = "ConnectedComponentOCRModule"

    def __init__(self, stage: int, parameters: Dict | None = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.mode = parameters.get('mode', 'Page')

        self.reference_mode = parameters.get('reference_mode', 'page')  # defines whether the new areas are referenced by the page or additionally by another area

        self.add_new_area = parameters.get(
            'add_new_area', ["OCRLine", "OCRString", "OCRChar"])

        self.ignore_overlap_merge_char = parameters.get(
            'ignore_overlap_merge_char', 3)
        self.ignore_overlap_merge_word = parameters.get(
            'ignore_overlap_merge_word', 3)

        self.exclude: Dict[str, List[str]] = parameters.get('exclude', {})

        self.page_range = parameters.get('page_range', None)
        self.character_distance = parameters.get('character_distance', 10)
        self.word_distance = parameters.get('word_distance', 20)
        self.min_width_bb = parameters.get('min_width_bb', 7)
        self.min_height_bb = parameters.get('min_height_bb', 10)
        self.max_size_bb = parameters.get('max_size_bb', 50)

        # avoid lines that are too high
        self.height_multiplier = parameters.get('height_multiplier', 2)
        self.initial_average_height = parameters.get(
            'initial_average_height', 20)

        self.skip_merge = parameters.get('skip_merge', False)

    def execute(self, inpt: Document) -> Document:
        if self.mode == 'Page':
            return self.execute_pages(inpt)
        else:
            ttt = self.execute_areas(inpt, self.mode)
            return ttt

    def execute_pages(self, inpt: Document) -> Document:
        # with parallelization
        with ProcessPoolExecutor() as executor:
            futures = list()
            for ix, p_id in enumerate(inpt.pages.allIds):
                if self.page_range is not None and ix not in self.page_range:
                    continue

                page_to_exclusion = dict()

                for k, exclude_area in self.exclude.items():
                    for ex in inpt.get_area_by(lambda x: x.category in exclude_area, page):
                        page_to_exclusion.setdefault(k, []).append(ex.boundingBox.get_in_img_space(
                            inpt.pages[page].factor_width, inpt.pages[page].factor_height).tuple())

                futures.append(executor.submit(self.execute_ocr, np.array(
                    inpt.get_img_page(p_id, as_string=False)), p_id, page_to_exclusion))

            for future in self.get_progress_bar(as_completed(futures), total=len(inpt.pages.allIds), unit="pages"):
                (lines, words_to_lines), (words,
                                          chars_to_words), chars, _, page = future.result()

                try:
                    chars_to_words = self.add_to_doc(
                        inpt, page, self.add_new_area[2], chars, None, chars_to_words)
                except Exception as e:
                    chars_to_words = None
                try:
                    words_to_lines = self.add_to_doc(
                        inpt, page, self.add_new_area[1], words, chars_to_words, words_to_lines)
                except Exception as e:
                    words_to_lines = None
                self.add_to_doc(
                    inpt, page, self.add_new_area[0], lines, words_to_lines, None)

        # print(f"Time: {time.time() - start_time}")

        # check if there are any strings that overlap
        # if so, merge them
        # import itertools
        # for page in inpt.pages.allIds:
        #     words = list(inpt.get_area_type("String", page))
        #     for word in itertools.combinations(words, 2):
        #         if word[0].boundingBox.overlap(word[1].boundingBox):
        #             print(f"Overlap {word[0].boundingBox} -- {word[1].boundingBox}  {word[0].boundingBox.intersection_over_union(word[1].boundingBox)}")

        return inpt

    def execute_areas(self, inpt: Document, category: str) -> Document:
        # currently only working for baseline
        xpadding = 10
        ypadding = 10
        areas = list(inpt.get_area_type(category))

        # drawn_lines cache
        page_to_drawn_lines = dict()

        for area in self.get_progress_bar(areas, unit="areas"):
            img = np.array(inpt.get_img_snippet(
                area.oid, padding=(xpadding, ypadding), as_string=False))
            page = inpt.find_page_of_area(area.oid)

            img_bb = area.boundingBox.get_in_img_space(inpt.pages[page].factor_width,
                                                        inpt.pages[page].factor_height)

            # get lines
            if page not in page_to_drawn_lines:
                page_to_exclusion = dict()

                for k, exclude_area in self.exclude.items():
                    for ex in inpt.get_area_by(lambda x: x.category in exclude_area, page):

                        bb = ex.boundingBox.get_in_img_space(
                            inpt.pages[page].factor_width, inpt.pages[page].factor_height)

                        if not bb.overlap(img_bb):
                            continue

                        x1, y1, x2, y2 = bb.tuple()

                        page_to_exclusion.setdefault(k, []).append((
                            x1-img_bb.x1, y1-img_bb.y1, x2-img_bb.x1, y2-img_bb.y1
                        ))
                page_to_drawn_lines[page] = page_to_exclusion
            exclude = page_to_drawn_lines[page]

            (lines, words_to_lines), (words,
                                      chars_to_words), chars, _, page = self.execute_ocr(img, page, exclude)

            # offset of original area has to be added to each bounding box
            def offset_bb(bb: BoundingBox):
                bb.x1 += img_bb.x1-xpadding
                bb.x2 += img_bb.x1-xpadding
                bb.y1 += img_bb.y1-ypadding
                bb.y2 += img_bb.y1-ypadding
                return bb

            def do_offset_bbs(bbs: List[BoundingBox]):
                # yield
                # for bb in bbs:
                #     yield offset_bb(bb)
                return [offset_bb(i) for i in bbs]

            try:
                chars_to_words = self.add_to_doc(inpt, page, self.add_new_area[2],
                                                 do_offset_bbs(chars), None, chars_to_words, referenced_by=area.oid)
            except Exception as e:
                chars_to_words = None
            try:
                words_to_lines = self.add_to_doc(inpt, page, self.add_new_area[1], do_offset_bbs(
                    words), chars_to_words, words_to_lines, referenced_by=area.oid)
            except Exception as e:
                words_to_lines = None
            self.add_to_doc(inpt, page, self.add_new_area[0], do_offset_bbs(
                lines), words_to_lines, None, referenced_by=area.oid)

        return inpt

    def add_to_doc(self, inpt: Document, p_id, kind, objs, sub_objs, self_objs, referenced_by: Optional[str] = None):
        sub_ids = [list() for _ in range(len(self_objs))
                   ] if self_objs is not None else None
        for ix, line in enumerate(objs):
            if sub_objs is not None and len(objs) == len(sub_objs):
                ids = inpt.add_area(p_id, kind, line, references=sub_objs[ix], referenced_by=referenced_by, convert_to_xml=False)
            else:
                ids = inpt.add_area(p_id, kind, line, referenced_by=referenced_by, convert_to_xml=False)
            if self_objs is not None:
                for ixx, sub_ix in enumerate(self_objs):
                    if ix in sub_ix:
                        try:
                            sub_ids[ixx].append(ids)
                        except:
                            sub_ids[ixx] = [ids]

        return sub_ids

    def execute_ocr(self, img: np.ndarray, page, exclude: Dict[str, List[Tuple]]) -> List[Area]:
        # make connected component
        import cv2
        import doxapy
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binarized = np.empty(gray.shape, dtype=np.uint8)

        # TODO: here should the dimensions be checked to be safe

        sauvola = doxapy.Binarization(doxapy.Binarization.Algorithms.SAUVOLA)
        sauvola.initialize(gray)
        sauvola.to_binary(binarized, parameters={
                          "window": 15, "k": 0.2, "R": 127})
        # expand to [:, :, 3]
        # binarized = np.repeat(binarized[:, :, np.newaxis], 3, axis=2)
        binarized = cv2.bitwise_not(binarized)

        (numLabels, _, stats, _) = cv2.connectedComponentsWithStats(
            binarized, connectivity=8)

        gray=None
        binarized=None
        img = None

        characters: List[BoundingBox] = list()
        drawn_lines = list()

        # first component is background
        for i in range(1, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # check if bounding box is in drawn lines
            tmp_bb = BoundingBox(x, y, x+w, y+h, img_sp=True)
            # if any([tmp_bb.overlap(i) for i in drawnlines]):
            #     continue
            if w > self.max_size_bb or h > self.max_size_bb:
                drawn_lines.append(tmp_bb)
            elif w < self.min_width_bb or h < self.min_height_bb:
                # print(f"skipped {w} {h}")
                continue
            else:
                characters.append(tmp_bb)

        # nms
        # len_before = len(characters)
        # characters = util.nms_merge(characters, 0.9)
        # self.debug_msg(f"nms merged from {len_before} to {len(characters)} boxes")

        # merge bounding boxes
        # strategy recursive
        # - iterate over every box
        # - go 'threshold' px to right
        # - if there is a box, merge it
        if not self.skip_merge:
            # do while change is true
            iterations = 0
            change = True

            average_height = np.median([x.height for x in characters])
            self.initial_average_height = average_height

            while change:
                words, map_c_to_w = merge_overlapping_bounding_boxes_in_one_direction(
                    characters, self.character_distance, -self.ignore_overlap_merge_char, exclude.get(self.add_new_area[1], []), direction=0, 
                    height_threshold=self.height_multiplier*self.initial_average_height)

                change = len(words) != len(characters)
                characters = words
                self.debug_msg(f"iteration {iterations} merged from {len(characters)} to {len(words)} boxes")
                iterations += 1
                if iterations > 10:
                    break
            words_ret = (words, map_c_to_w)

            change = True
            iterations = 0
            while change:
                lines, map_w_to_l = merge_overlapping_bounding_boxes_in_one_direction(
                    words, self.word_distance, -self.ignore_overlap_merge_word, exclude.get(self.add_new_area[0], []), direction=0, height_threshold=self.height_multiplier*self.initial_average_height)

                change = len(lines) != len(words)
                words = lines
                self.debug_msg(f"iteration {iterations} merged from {len(words)} to {len(lines)} boxes")
                iterations += 1
                if iterations > 10:
                    break
            lines_ret = (lines, map_w_to_l)

            return lines_ret, words_ret, characters, drawn_lines, page,
        else:
            return ([], {}), ([], {}), characters, drawn_lines, page
