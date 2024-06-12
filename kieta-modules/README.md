# KIETA Modules

## Modules used within the pipeline

### PDFConvert
This module is responsible for converting PDF files into the internal data structure defined by kieta-data-objs.

### TATRDetection (tatr.py)
Performs Table Detection (TD) to detect tables in documents.
This module is based on 
```
@software{smock2021tabletransformer,
  author = {Smock, Brandon and Pesala, Rohith},
  month = {06},
  title = {{Table Transformer}},
  url = {https://github.com/microsoft/table-transformer},
  version = {1.0.0},
  year = {2021}
}
```

### ConnectedComponentOCRModule (ocr/text_detection.py)
Performs connected component analysis to detect text regions.

### GuppyOCRModule (ocr/ocr.py)
Performs ocr on specified regions.

### CellDetector (cellDetector.py)
Detects cells in specified regions.

### MarkovChainRecognizer (heuristicRecognition.py)
Performs Table Structure Recognition (TSR) to arrange cells as table matrix.

### FunctionalRecognizer (functionalRecognition.py)
Performs functional analysis of table cells.

### MetadataExtractor (specific_modules/glass/meta_data.py)
Performs metadata lookup for specified patent number.

### GlassExtractor (specific_modules/glass/glassExtraction.py)
Extracts glass configurations across tables. Simple recognition of units included.



<!-- ### Export

- ExportPageXMLModule
    - confidence_threshold (default: -1)
        - The confidence threshold for exporting text elements. Text elements with a confidence score below this threshold will not be exported.
    - text_category (default: "String")
        - Lowest category containing textual content
    - line_category (default: "Line")
        - One hierarchical level above the text_category
    - optimize_baseline (default: False)
        - Whether to optimize the baseline of exported text elements.
    - bb_pad (default: 0)
        - The padding value for bounding boxes of exported elements.
- ExportJSONModule
    - include_pictures (default: False)
        - Whether to export pictures or not
- ExportTablesModule
    - table_category_name (default: "Table")
        - The category name for exported table elements.
- ExportTextModule
    - strings (default: "String")
        - Lowest category containing textual content
    - lines (default: "Line")
        - One hierarchical level above the text_category
- ExportImageModule
    - category (default: "Page")
        - Which category to export
- ExportMaskedImageModule
    - category (default: ["String"])
        - Which categories to export
    - color (default: ["red"])
        - Defines boundingbox color of each category. Has to be same length as category
- ExportCOCOModule
    - super_categories (default: ["Table"])
    - type (default: ["Table"]) -->

