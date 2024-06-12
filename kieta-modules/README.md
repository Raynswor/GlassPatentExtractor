# Kieta Modules


## Available Modules

### Import

#### PDFConvert
#### PageXMLImport
#### JSONImport
#### ImageImport
## Available Modules

### Import

- PDFConvert
- PageXMLImport
- JSONImport
- ImageImport

### Export

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
    - type (default: ["Table"])

