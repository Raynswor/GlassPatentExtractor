from kieta_modules import Module, util

from typing import Any, Dict, Generator, Iterable, List, Tuple

from kieta_data_objs import Document, BoundingBox, Area

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

from PIL import Image

class GuppyOCRModule(Module):
    _MODULE_TYPE="GuppyOCRModule"

    def __init__(self, stage: int, parameters: Dict | None = None, debug_mode: bool = False) -> None:
        from guppyocr.api import GuppyOCR
        
        super().__init__(stage, parameters, debug_mode)
        model_path = parameters.get('model_path', "")
        if model_path is None:
            raise ValueError('No model path provided for OCR model')

        self.model = GuppyOCR.load_model(model_path, device=parameters.get('device', 'cuda'))
        # categories to be processed by this module
        self.apply_to = parameters.get('apply_to', ['OCRLine'])

        # string, line
        self.add_new_area = parameters.get('add_new_area', None)
        self.split_words = parameters.get('split_words', False)

        self.batch_size = parameters.get('batch_size', 16)

        assert self.split_words == False or len(self.add_new_area) == 2, "split_words only works with add_new_area='String'"

        self.padding = parameters.get('padding', (0,0))  # (x,y)

    def execute(self, inpt: Document) -> Document:
        # multiprocess futures
        with ThreadPoolExecutor() as executor:
            futures = list()
            for p_id in inpt.pages.allIds:
                # number_areas_per_page.append(len(list(inpt.get_area_by(lambda x: x.category in self.apply_to, page=p_id))))
                futures.append(executor.submit(self.predict_multiple_snippets_from_one_img, 
                                            np.array(inpt.get_img_page(p_id, as_string=False)), 
                                            inpt.get_area_by(lambda x: x.category in self.apply_to, page=p_id)
                                            )
                                        )
            
            for future in self.get_progress_bar(as_completed(futures), total=len(inpt.pages.allIds), unit="pages"):
                counter = 0
                for oid, text, conf in future.result():
                    counter += 1

                    # if counter > number_areas_per_page[int(oid.split('-')[1])]:
                    #     self.error_msg = f"More areas than expected for page {oid.split('-')[1]}"
                    #     break

                    area = inpt.get_area_obj(oid)
                    p_id = f"page-{oid.split('-')[1]}"
                    if self.add_new_area:
                        if self.split_words:
                            curr = []
                            # calculate bounding box for each word
                            if not text or len(text) == 0:
                                char_width = 0
                            else:
                                try:
                                    char_width = area.boundingBox.width / len(text)
                                except ZeroDivisionError:
                                    char_width = 0
                            for word in text.split(' '):
                                if not word:
                                    continue
                                idx = text.index(word)
                                if char_width != 0:
                                    x1 = area.boundingBox.x1 + char_width * idx
                                    x2 = x1 + char_width * len(word)
                                else:
                                    x1 = area.boundingBox.x1
                                    x2 = area.boundingBox.x2
                                curr.append(inpt.add_area(p_id,
                                            self.add_new_area[-1], 
                                            BoundingBox(x1, area.boundingBox.y1, x2, area.boundingBox.y2, img_sp=area.boundingBox.img_sp),
                                            data={'content': word.strip()}, 
                                            confidence=conf,
                                            id_prefix="Gup", convert_to_xml=False))
                            inpt.add_area(p_id, self.add_new_area[0], area.boundingBox, references=curr, id_prefix="Gup")
                        else:
                            inpt.add_area(p_id,
                                        self.add_new_area, 
                                        area.boundingBox, 
                                        data={'content': text.strip()}, 
                                        confidence=conf,
                                        id_prefix="Gup", convert_to_xml=False)
                    else:
                        area.data['content'] = text.strip()
                        area.confidence = conf
                        # Image.fromarray(img).save(f"/tmp/test/{area.oid}.jpg")
                        # pass

        return inpt
    
    def predict_multiple_snippets_from_one_img(self, img: np.ndarray, areas: Iterable[Area]) -> Generator[str, str, float]:
        # for area in areas:
            # try:
            #     yield area.oid, *self.model.ocr_with_confidence(
            #         img[
            #             max(int(area.boundingBox.y1) - self.padding[1], 0) : min(int(area.boundingBox.y2) + self.padding[1], img.shape[0]-1),
            #             max(int(area.boundingBox.x1) - self.padding[0], 0) : min(int(area.boundingBox.x2) + self.padding[0], img.shape[1]-1),
            #             ::-1,
            #         ]
            #     )
            # except Exception as e:
            #     print(e)
            #     yield area.oid, "", 0.0
        

        # do in batches of 8
        areas = list(areas)
        for i in range(0, len(areas), 8):
            ret = self.model.ocr_with_confidence_batch([img[
                    max(int(area.boundingBox.y1) - self.padding[1], 0) : min(int(area.boundingBox.y2) + self.padding[1], img.shape[0]-1),
                    max(int(area.boundingBox.x1) - self.padding[0], 0) : min(int(area.boundingBox.x2) + self.padding[0], img.shape[1]-1),
                    ::-1,
            ] for area in areas[i:i+8]], self.batch_size)
            for area, (text, conf) in zip(areas[i:i+8], ret):
                yield area.oid, text, conf

        # ret = self.model.batch_ocr_with_confidence([img[
        #             max(int(area.boundingBox.y1) - self.padding[1], 0) : min(int(area.boundingBox.y2) + self.padding[1], img.shape[0]-1),
        #             max(int(area.boundingBox.x1) - self.padding[0], 0) : min(int(area.boundingBox.x2) + self.padding[0], img.shape[1]-1),
        #             ::-1,
        # ] for area in areas])
        # for area, (text, conf) in zip(areas, ret):
        #     yield area.oid, text, conf


class TesseractOCRModule(Module):
    _MODULE_TYPE="TesseractOCRModule"

    """
    Adds: OCRed lines and strings within
    """

    def __init__(self, stage: int, parameters: Dict | None = None, debug_mode: bool = False) -> None:
        from pytesseract import image_to_data
        super().__init__(stage, parameters, debug_mode)
        self.mode = parameters.get('mode', 'Page')
        self.lang = parameters.get('lang', 'eng')

        self.add_new_area = parameters.get('add_new_area', ["OCRLine", "OCRString"])

        self.processing_method = image_to_data


    def execute(self, inpt: Document) -> Document:
        if self.mode == 'Page':
            return self.execute_pages(inpt)
        else:
            ttt = self.execute_areas(inpt, self.mode)
            return ttt

    def execute_pages(self, inpt: Document) -> Document:
        # with ProcessPoolExecutor() as executor:
        #     futures = list()
        #     for p_id in inpt.pages.allIds:
        #         futures.append(executor.submit(self.extract_text_areas, inpt.get_img_page(p_id, as_string=False)))
            
        #     for future, p_id in self.get_progress_bar(zip(as_completed(futures), inpt.pages.allIds), total=len(inpt.pages.allIds), unit="pages"):
        #         lines, ltoa_map = future.result()
        #         for (line, areas) in zip(lines, ltoa_map.values()):
        #             ref = list()
        #             for a in areas:
        #                 ref.append(inpt.add_area(p_id, 'OCRString', a.boundingBox, data=a.data))
        #             inpt.add_area(p_id, 'OCRLine', line.boundingBox, references=ref)

        # without parallelization
        for p_id in self.get_progress_bar(inpt.pages.allIds, unit="pages"):
            # delete all other areas of type line and string
            lines, ltoa_map = self.extract_text_areas(inpt.get_img_page(p_id, as_string=False))
            for (line, areas) in zip(lines, ltoa_map.values()):
                ref = list()
                for a in areas:
                    ref.append(inpt.add_area(p_id, self.add_new_area[1], a.boundingBox, data=a.data, confidence=a.confidence, id_prefix="Tes"))
                inpt.add_area(p_id, self.add_new_area[0], line.boundingBox, references=ref, id_prefix="Tes")
        return inpt
    
    def execute_areas(self, inpt: Document, category: str) -> Document:
        # currently only working for baseline
        temp = list(inpt.get_area_type(category))
        for area in self.get_progress_bar(temp, unit="areas"):
            img = inpt.get_img_snippet(area.oid, as_string=False)
            lines, ltoa_map = self.extract_text_areas(img)
            for (line, areas) in zip(lines, ltoa_map.values()):
                ref = list()
                for a in areas:
                    ref.append(inpt.add_area(f"page-{area.oid[0]}",  self.add_new_area[1], a.boundingBox, data=a.data, confidence=a.confidence, id_prefix="Tes"))
                inpt.add_area(f"page-{area.oid[0]}",  self.add_new_area[0], line.boundingBox, references=ref, id_prefix="Tes")
        return inpt
    
    def execute_areas_V2(self, inpt: Document, category: str) -> Document:
        with ThreadPoolExecutor() as executor:
            # currently only working for baseline
            futures = []
            for p_id in inpt.pages.allIds:
                for area in inpt.get_area_type(category, page=p_id):
                    img = inpt.get_img_snippet(area.oid, as_string=False)
                    futures.append(executor.submit(self.extract_text_areas, img))
            for future, p_id in zip(as_completed(futures), inpt.pages.allIds):
                lines, ltoa_map = future.result()
                for (line, areas) in zip(lines, ltoa_map.values()):
                    ref = list()
                    for a in areas:
                        ref.append(inpt.add_area(p_id,  self.add_new_area[1], a.boundingBox, data=a.data, confidence=a.confidence))
                    inpt.add_area(p_id,  self.add_new_area[0], line.boundingBox, references=ref)
        return inpt
        

    def extract_text_areas(self, img):
        data = self.processing_method(img, 
                                         output_type='data.frame',
                                         lang=self.lang,)
        blocks, lines_areas_map = dict(), dict()
        # level   page_num        block_num       par_num line_num        word_num        left    top     width   height  conf    text
        for st in data.index:
            # print(f"st  {data['text'][st]}, {type(data['text'][st])},  {data['text'][st] == np.nan}")
            if data['conf'][st] < 0 or not data['text'][st]:
                continue
            try:
                if np.isnan(data['text'][st]):
                    continue
            except:
                if data['text'][st].isspace():
                    continue
            # make area
            area = Area("", 'OCRString', 
                                    BoundingBox(data['left'][st], data['top'][st], data['left'][st]+data['width'][st], data['top'][st]+data['height'][st], img_sp=True),
                                    data={'content': str(data['text'][st]).strip()}, confidence=float(data['conf'][st]))
            if data['block_num'][st] not in blocks:
                blocks[data['block_num'][st]] = [area]
            else:
                blocks[data['block_num'][st]].append(area)
            if data['line_num'][st] not in lines_areas_map:
                lines_areas_map[data['line_num'][st]] = [area]
            else:
                lines_areas_map[data['line_num'][st]].append(area)
        lines = list()
        for k, l in lines_areas_map.items():
            # merge bb
            bb: BoundingBox = l[0].boundingBox
            try:
                for i in l[1:]:
                    bb = bb.__expand__(i.boundingBox)
            except IndexError:
                pass
            bb.img_sp = True
            lines.append(Area(k, 'OCRLine', bb))
        return lines, lines_areas_map


class PaddleOCRModule(Module):
    _MODULE_TYPE="PaddleOCRModule"

    def __init__(self, stage: int, parameters: Dict | None = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.mode = parameters.get('mode', 'Page')
        self.lang = parameters.get('lang', 'en')
        self.add_new_area = parameters.get('add_new_area', ["OCRLine", "OCRString"])

        from paddleocr import PaddleOCR,draw_ocr
        # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
        # You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
        # to switch the language model in order.
        self.ocr = PaddleOCR(use_angle_cls=False, lang=self.lang) # need to run only once to download and load model into memory
    
    def execute(self, inpt: Document) -> Document:
        if self.mode == 'Page':
            return self.execute_pages(inpt)
        else:
            ttt = self.execute_areas(inpt, self.mode)
            return ttt
    
    def execute_pages(self, inpt: Document) -> Document:
        for p_id in self.get_progress_bar(inpt.pages.allIds, unit="pages"):
            # delete all other areas of type line and string
            lines = self.execute_ocr(np.array(inpt.get_img_page(p_id, as_string=False)))
            for line in lines:
                st = inpt.add_area(p_id, self.add_new_area[1], line.boundingBox, data=line.data, confidence=line.confidence)
                inpt.add_area(p_id, self.add_new_area[0], line.boundingBox, references=[st])

        return inpt
    

    def execute_ocr(self, img):    
        result = self.ocr.ocr(img, cls=False)

        lines = list()

        for idx in range(len(result)):
            line = result[idx]
            boxes = line[0]
            txts = line[1][0].strip()
            scores = line[1][1]
            print(boxes, txts, scores)
            lines.append(Area(idx, 'OCRLine', BoundingBox(boxes[0][0], boxes[0][1], boxes[2][0], boxes[2][1], img_sp=True),
                                data={'content': txts}, confidence=scores))

        return lines
