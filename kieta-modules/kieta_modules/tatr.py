import itertools
from typing import List, Optional, Dict, Tuple

import numpy as np
from kieta_data_objs import Document, BoundingBox
from kieta_data_objs.util import base64_to_img
from kieta_modules import Module
from kieta_modules.util import get_overlapping_areas, nms_merge
import torch

from transformers import DetrImageProcessor,TableTransformerForObjectDetection, TableTransformerForObjectDetection, DetrFeatureExtractor

import sys
sys.path.append("./utility_stuff/tatr/detr")
from .utility_stuff.tatr.detr.models import build_model
from .utility_stuff.tatr.detr.util.misc import NestedTensor

import json

from dataclasses import dataclass
from torchvision import transforms

from fitz import Rect


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def outputs_to_objects(outputs, img_size, class_idx2name):
    m = outputs['pred_logits'].softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = class_idx2name[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects


def objects_to_crops(img, tokens, objects, class_thresholds, padding=10):
    """
    Process the bounding boxes produced by the table detection model into
    cropped table images and cropped tokens.
    """
    def iob(bbox1, bbox2):
        """
        Compute the intersection area over box area, for bbox1.
        """
        intersection = Rect(bbox1).intersect(bbox2)
        
        bbox1_area = Rect(bbox1).get_area()
        if bbox1_area > 0:
            return intersection.get_area() / bbox1_area
        
        return 0

    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}

        bbox = obj['bbox']
        bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]

        table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
        for token in table_tokens:
            token['bbox'] = [token['bbox'][0]-bbox[0],
                             token['bbox'][1]-bbox[1],
                             token['bbox'][2]-bbox[0],
                             token['bbox'][3]-bbox[1]]

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [cropped_img.size[0]-bbox[3]-1,
                        bbox[0],
                        cropped_img.size[0]-bbox[1]-1,
                        bbox[2]]
                token['bbox'] = bbox

        cropped_table['tokens'] = table_tokens

        table_crops.append(cropped_table)

    return table_crops


class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
        
        return resized_image

detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@dataclass
class Temp:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


class TATRDetection(Module):
    _MODULE_TYPE = 'TATRDetection'

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.feature_extractor = DetrFeatureExtractor()
        # self.feature_extractor = DetrImageProcessor()
        self.parameters['threshold'] = float(self.parameters.get('threshold', 0.5))
        self.parameters['nms_threshold'] = float(self.parameters.get('nms_threshold', 0.5))
        # shrinks/extends bounding box to included DrawnLine areas
        self.parameters['adjust_to'] = self.parameters.get('adjust_to', [])
        self.parameters['iterations'] = int(self.parameters.get('iterations', 1))
        self.parameters['only_captions'] = self.parameters.get('only_captions', False)

        self.det_device = self.parameters.get("device", "cuda")
        # self.model = TableTransformerForObjectDetection.from_pretrained("bsmock/TATR-v1.1-All")
        # TODO: UPDATE TO NEW MODEL
        # self.model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        if self.parameters.get("config", None):
            with open(self.parameters.get("config", None), 'r') as f:
                det_config = json.load(f)
            det_args = type('Args', (object,), det_config)
            det_args.device = self.det_device
            self.model, _, _ = build_model(det_args)
            print("Detection model initialized.")

            self.model.load_state_dict(torch.load(self.parameters.get("model", ""), map_location=torch.device(self.det_device)))
            self.model.to(self.det_device)
            self.model.eval()
            print("Detection model weights loaded.")

            self.MODE = "DETR"
        else:
            self.MODE = None
            self.model = TableTransformerForObjectDetection.from_pretrained(self.parameters.get("model", "microsoft/table-transformer-detection"))

    def execute(self, inpt: Document) -> Document:
        for page_id in self.get_progress_bar(inpt.pages.allIds, unit="pages"):
            captions = list()
            if self.parameters['only_captions']:
                for area_id in inpt.references[page_id]:
                    if inpt.areas[area_id].category == 'Caption':
                        captions.append(area_id)
                if len(captions) == 0:
                    continue

            image = base64_to_img(inpt.pages[page_id].img)
            width, height = image.size
            
            if self.MODE == "DETR":
                encoding = detection_transform(image)
            else:  # hugging face
                encoding = self.feature_extractor(image, return_tensors="pt")       

            results = {
                "scores": [],
                "boxes": [],
            }
            with torch.no_grad():
                for it in range(self.parameters['iterations']):
                    res = dict()
                    if self.MODE == "DETR":
                        outputs = self.model([encoding.to(self.det_device)])
                        outputs = Temp(outputs['pred_logits'], outputs['pred_boxes'])
                    else:  # hugging face
                        outputs = self.model(**encoding)
                    res = self.feature_extractor.post_process_object_detection(outputs, threshold=self.parameters['threshold'], target_sizes=[(height, width)])[0]
                    results['scores'].extend(res['scores'].tolist())
                    results['boxes'].extend(res['boxes'].tolist())
                    self.debug_msg(f"Iteration {it}: {len(res['scores'])} tables on {page_id}")

            drawn_stuff = list()
            if self.parameters['adjust_to']:
                for area_type in self.parameters['adjust_to']:
                    drawn_stuff.extend(inpt.get_area_type(area_type, page=page_id))

            tabs: List[Tuple[BoundingBox, float]] = list()
            scores: List[float] = list()
            for tab_idx, (score, bb) in enumerate(zip( results['scores'], results['boxes'])):
                bb = BoundingBox(*bb, img_sp=True)
                if self.parameters['adjust_to']:
                    try:
                        lines = [x for y in get_overlapping_areas(bb, drawn_stuff, None, factors=(
                            inpt.pages.byId[page_id].factor_width, inpt.pages.byId[page_id].factor_height
                        )).values() for x in y ]
                        self.debug_msg(f"Found {len(lines)} lines in table {tab_idx}")
                        # extend table bounding box to width and height of lines
                        # find min max of lines
                        min_x, min_y, max_x, max_y = 999999, 999999, 0, 0
                        for l in lines:
                            # if horizontal
                            if l.boundingBox.is_horizontal():
                                min_x = min(min_x, l.boundingBox.x1)
                                max_x = max(max_x, l.boundingBox.x2)
                            else:
                                min_y = min(min_y, l.boundingBox.y1)
                                max_y = max(max_y, l.boundingBox.y2)
                        
                        self.debug_msg(f"Adjusted table bounding box by {min_x-bb.x1}, {min_y-bb.y1}, {max_x-bb.x2}, {max_y-bb.y2}")
                        bb.x1 = min_x if min_x != 999999 else bb.x1
                        bb.y1 = min_y if min_y != 999999 else bb.y1
                        bb.x2 = max_x if max_x != 0 else bb.x2
                        bb.y2 = max_y if max_y != 0 else bb.y2
                    except KeyError:
                        pass
                # add table area
                tabs.append(bb)
                scores.append(score)

            # nms
            bbls = list()
            used_captions = set()
            # zip two lists together
            tmp = list(zip(*nms_merge(tabs, self.parameters['nms_threshold'], scores)))
            for idx, (bb, score) in enumerate(sorted(tmp, key=lambda x: x[0].y1)):
                # try to find closest caption and extract numbering from it
                if self.parameters['only_captions']:
                    closest = None
                    closest_id = None
                    closest_dist = 999999
                    closest_number = None
                    for cap in captions:
                        cc = inpt.areas.byId[cap].boundingBox.get_in_img_space(inpt.pages.byId[page_id].factor_width, inpt.pages.byId[page_id].factor_height)
                        # horizontal overlap required
                        if not bb.overlap_horizontally(cc):
                            continue
                        dist = cc.distance_vertical(bb)
                        if dist < closest_dist:
                            closest_id = cap
                            closest = cc
                            closest_dist = dist
                            closest_number = inpt.areas.byId[cap].data['number']
                            closest_continued = inpt.areas.byId[cap].data.get('continued', False)
                    if closest is not None and closest_id not in used_captions:
                        # extend till it is near the caption
                        if bb.y1 > closest.y2:
                            bb.y1 = closest.y2+5
                        elif bb.y2 < closest.y1:
                            bb.y2 = closest.y1-5
                        inpt.add_area(page_id, 'Table', bb, data={'number': closest_number, 
                                                                  'continued': closest_continued,
                                                                  'caption': closest_id}, confidence=float(score), convert_to_xml=False)
                        used_captions.add(closest_id)
                    elif closest is not None and closest_id in used_captions:
                        # extend till it is near the caption
                        if bb.y1 > closest.y2:
                            bb.y1 = closest.y2+5
                        elif bb.y2 < closest.y1:
                            bb.y2 = closest.y1-5
                        # check if it overlaps with another table
                        overlap = None
                        for b in inpt.get_area_type('Table', page=page_id):
                            if b.boundingBox.overlap(bb):
                                overlap = b
                                break
                        if not overlap:
                            inpt.add_area(page_id, 'Table', bb, data={'number': closest_number, 
                                                                  'continued': closest_continued,
                                                                  'caption': closest_id}, confidence=float(score), convert_to_xml=False)
                        else:
                            # merge with other table
                            overlap.boundingBox = overlap.boundingBox.__expand__(bb)
                    else:
                        inpt.add_area(page_id, 'Table', bb, data={'number': idx}, confidence=float(score), convert_to_xml=False)
                else:
                    inpt.add_area(page_id, 'Table', bb, data={'number': idx}, confidence=float(score), convert_to_xml=False)
                bbls.append(bb)
        return inpt


class TATRRecognition(Module):
    _MODULE_TYPE = 'TATRRecognition'

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.feature_extractor = DetrFeatureExtractor()
        self.model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")

    def execute(self, inpt: Document) -> Document:
        for table in list(inpt.get_area_type('Table')):
            page_id = inpt.find_page_of_area(table.oid)

            included_areas = list()
            try:
                for ref in inpt.references.byId[table.oid]:
                    if inpt.areas.byId[ref].category == 'TableCell':
                        included_areas.append(inpt.areas.byId[ref])
            except KeyError:
                for aa in inpt.get_area_type('String', page=page_id):
                    if inpt.areas.byId[table.oid].boundingBox.overlap(aa.boundingBox):
                        included_areas.append(aa)
                    # subtract table x1, y1
            words = [{
                'bbox':
                    [aa.boundingBox.x1 - table.boundingBox.x1,
                        aa.boundingBox.y1 - table.boundingBox.y1,
                        aa.boundingBox.x2 - table.boundingBox.x1,
                        aa.boundingBox.y2- table.boundingBox.y1],
                'text': aa.data['content'],
                'span_num': idx,
                'line_num': 0,
                'block_num': 0,
            }  for idx, aa in enumerate(included_areas)]


            image = inpt.get_img_snippet(table.oid, as_string=False)
            encoding = self.feature_extractor(image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**encoding)
            
            target_sizes = [image.size[::-1]]
            results = self.feature_extractor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]

            st_to_idx = {
                'table': 0,
                'table column': 1,
                'table row': 2,
                'table column header': 3,
                'table projected row header': 4,
                'table spanning cell': 5,
                'no object': 6
            }
            str_class_idx2name = {v:k for k, v in st_to_idx.items()}
            tab_xml_bb = table.boundingBox.get_in_img_space(inpt.pages.byId[page_id].factor_width, inpt.pages.byId[page_id].factor_height)

            for r in zip(results['scores'], results['labels'], results['boxes']):
                # print(r)
                bbox = r[2].numpy()
                bbox = BoundingBox(
                                bbox[0] + tab_xml_bb.x1,
                                bbox[1] + tab_xml_bb.y1,
                                bbox[2] + tab_xml_bb.x1,
                                bbox[3] + tab_xml_bb.y1,
                                img_sp=True)
                inpt.add_area(page_id, str_class_idx2name[r[1].item()], bbox, confidence=r[0].item())

            # objects = outputs_to_objects(outputs, image.size, str_class_idx2name)

            # # Further process the detected objects so they correspond to a consistent table
            # tables_structure = objects_to_structures(objects, words, {
            #                                                         "table": 0.5,
            #                                                         "table column": 0.5,
            #                                                         "table row": 0.5,
            #                                                         "table column header": 0.5,
            #                                                         "table projected row header": 0.5,
            #                                                         "table spanning cell": 0.5,
            #                                                         "no object": 10
            #                                                     })

            # # Enumerate all table cells: grid cells and spanning cells
            # tables_cells = structure_to_cells(tables_structure[0], words)[0]
            # tab_xml_bb = table.boundingBox.get_in_xml_space(inpt.pages.byId[page_id].factor_width, inpt.pages.byId[page_id].factor_height)

            # # tables_cells = self.feature_extractor.post_process_object_detection(outputs, threshold=0.5, target_sizes=[image.size])[0]['boxes']

            # print(f"Found {len(tables_cells)} cells in table {table.oid}")
            # for cell in tables_cells:
            #     # add table x1,y1
            #     # print(cell['bbox'])
            #     cell['bbox'] = BoundingBox(
            #                     cell['bbox'][0] + tab_xml_bb.x1,
            #                     cell['bbox'][1] + tab_xml_bb.y1,
            #                     cell['bbox'][2] + tab_xml_bb.x1,
            #                     cell['bbox'][3] + tab_xml_bb.y1,
            #                     img_sp=False)
            #     # cell['bbox'] = BoundingBox(
            #     #                 cell[1] + tab_xml_bb.y1,
            #     #                 cell[2] + tab_xml_bb.x1,
            #     #                 cell[3] + tab_xml_bb.y1,
            #     #                 cell[0] + tab_xml_bb.x1,
            #     #                 img_sp=False)
            #     # print(f"after {cell['bbox']}")
            # if len(tables_cells) > 0:
            #     num_columns = max(
            #         [max(cell['column_nums']) for cell in tables_cells]) + 1
            #     num_rows = max([max(cell['row_nums']) for cell in tables_cells]) + 1
            # else: continue

            # table_array = np.empty([num_rows, num_columns], dtype="object")
            # table_array[:] = None
            # # for ic, col in enumerate(tables_cells):
            # #     for ir, cell in enumerate(col):

            # for cell in tables_cells:
            #     cell_id = inpt.add_area(page_id, 'TableCell',
            #                                 cell["bbox"],
            #                                 referenced_by=table.oid,
            #                                 data={'content': cell['cell text']})
            #     for (rr, cc) in itertools.product(cell['row_nums'], cell['column_nums']):
            #         table_array[rr][cc] = cell_id
            # inpt.areas.byId[table.oid].data['cells'] = table_array.tolist()
        return inpt