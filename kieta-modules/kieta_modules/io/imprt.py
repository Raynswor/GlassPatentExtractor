

import json
import os
from typing import Dict, List, Optional, Tuple

from kieta_data_objs import Document, NormalizedObj, Page, Revision, Area, BoundingBox
from kieta_data_objs.util import base64_to_img

from .. import Module, util

from .pdfConvert import PDFConverter

import cv2


class ImportSwitch(Module):
    _MODULE_TYPE = 'ImportSwitch'

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.importers = {
            "JSON": JSONImport,
            "PageXML": PageXMLImport,
            "Image": ImageImport,
            "DIWTable": DIWTableImport,
            "PDF": PDFConverter
        }

    def execute(self, inpt: Dict) -> Document:
        """
        Loads either a file or an xml string as KIETA document
        """
        # check if file
        if isinstance(inpt, dict):
            pass
        else:
            raise ValueError("Input type not supported")
        # get the type of the input
        inpt_type = None
        match inpt.get('suffix', '').replace('.', '').lower():
            case "json":
                loaded = json.loads(inpt['file'])
                if "references" in loaded:
                    inpt_type = "JSON"
                else:
                    inpt_type = "DIWTable"
            case "xml":
                inpt_type = "PageXML"
            case "png", "jpg":
                inpt_type = "Image"
            case "pdf":
                inpt_type = "PDF"
        if inpt_type is None:
            raise ValueError("Input type not supported")

        # get the importer
        importer = self.importers[inpt_type](
            self.stage, self.parameters, self.debug_mode)
        return importer.execute(inpt)


class PageXMLImport(Module):
    _MODULE_TYPE = 'PageXMLImport'

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.dpi = int(parameters.get("dpi", 300))

    def execute(self, inpt: str) -> Document:
        """
        Loads either a file or an xml string as KIETA document
        """
        # check if file
        if os.path.isfile(inpt):
            with open(inpt, "r") as f:
                xml = f.read()
        else:
            xml = inpt

        # load xml


class JSONImport(Module):
    _MODULE_TYPE = 'JSONImport'

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)

    def execute(self, inpt: str) -> Document:
        """
        Loads json string as KIETA document
        """
        import jsonpickle
        # check if file

        if isinstance(inpt, dict):
            if "file" in inpt:
                json = jsonpickle.decode(inpt['file'])
            else:
                json = inpt
        else:
            try:
                json = jsonpickle.decode(inpt)
            except:
                json = inpt

        # load json
        doc = Document.from_dic(json)
        return doc


class ImageImport(Module):
    _MODULE_TYPE = 'ImageImport'

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)

    def execute(self, inpt: str) -> Document:
        """
        Loads either a file or a json string as KIETA document
        """
        # create document
        doc = Document(
            oid=inpt.split("/")[-1].removesuffix(".png").removesuffix(".jpg"),
            pages=NormalizedObj({}, []),
            areas=NormalizedObj({}, []),
            revisions=[],
            metadata={},
            references=NormalizedObj({}, [])
        )
        doc.add_revision("initial")

        # check if file
        img = cv2.imread(inpt)
        doc.add_page(img=img)
        return doc


class DIWTableImport(Module):
    _MODULE_TYPE = 'DIWTableImport'

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)

    def execute(self, inpt: Dict) -> Document:
        """
        Loads either a file or a json string as KIETA document
        """
        # read the data from json and add it to the document

        self.info_msg(f"TYPE {type(inpt)}")

        import jsonpickle
        if isinstance(inpt, str):
            data = jsonpickle.decode(inpt)
            self.info_msg(list(data.keys()))
        elif isinstance(inpt, dict):
            data = jsonpickle.decode(inpt['file'])
        else:
            data = inpt

        # create document
        doc = Document(
            oid=data.get('oid', inpt.get('id')),
            pages=NormalizedObj(),
            areas=NormalizedObj(),
            revisions=[],
            metadata={},
            references=NormalizedObj()
        )
        doc.add_revision("initial")

        doc.add_page(Page("page-0", 0, ""))

        page_img = None

        try:
            del data['oid']
        except:
            pass
        table_id = None
        for area in data.values():
            if area['category'] == "Table":
                table_id = area['oid']
                page_img = area['data'].get('page_img', None)

                doc.add_area("page-0",
                             area['category'],
                             BoundingBox(**area['boundingBox']), data=area['data'], confidence=area['confidence'], area_id=area['oid'])
            else:
                doc.add_area("page-0", area['category'], self.extract_from_polyline(area['boundingBox']),
                             data=area['data'], confidence=area['confidence'], referenced_by=table_id, area_id=area['oid'])

        if page_img is not None:
            doc.pages["page-0"].img = page_img
            # convert to img
            pic = base64_to_img(page_img)
            doc.pages["page-0"].xml_width = pic.width
            doc.pages["page-0"].xml_height = pic.height
            doc.pages["page-0"].img_width = pic.width
            doc.pages["page-0"].img_height = pic.height
            doc.pages["page-0"].factor_width = 1
            doc.pages["page-0"].factor_height = 1

        return doc

    def extract_from_polyline(self, line: List[Tuple[int, int]]):
        # Extract x and y coordinates
        try:
            x_coords, y_coords = zip(*line)
        except:
            return BoundingBox(0, 0, 1, 1, img_sp=True)

        # Find min and max coordinates
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        return BoundingBox(min_x, min_y, max_x, max_y, img_sp=True)
