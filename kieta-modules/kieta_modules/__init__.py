
from .utility_stuff.tatr import detr as detr
from .utility_stuff.layoutlm import layoutlm as layoutlm

from .base import Module
from .pipeline import Pipeline, PipelineManager
# from .camelotTable import CamelotDetector
from .heuristicDetection import HeuristicDetector
from .captionDetector import CaptionDetector
from .captionTableLocalizer import CaptionTableLocalizer
from .cellDetector import CellDetector
from .heuristicRecognition import HeuristicRecognizer, MarkovChainRecognizer
from .io.pdfConvert import PDFConverter
from .functionalRecognition import FunctionalRecognizer
from .tatr import TATRDetection, TATRRecognition
from .layout.paragraphDetection import ParagraphDetector
from .layout.lineDetection import LineDetectorCanny, LogicalLineGuesser
from .io.exprt import ExportImageModule, ExportMaskedImageModule, \
ExportJSONModule, ExportPageXMLModule, ExportTextModule, \
ExportTableContentModule,ExportTableStructureModule, ExportCOCOModule, ExportWordsModule, ExportMaskedSVGModule
from .io.imprt import PageXMLImport, JSONImport, ImageImport, DIWTableImport, ImportSwitch

from .utilityModules import UtilityModule, DeleteUtilityModule

from .ocr.text_detection import BaselineDetector, ConnectedComponentOCRModule
from .ocr.ocr import GuppyOCRModule, TesseractOCRModule, PaddleOCRModule
from .ocr.merge import MergeOCRModule
from .ocr.post_correction import DictionaryCorrectorModule


# specific modules
from .specific_modules import *
from .specific_modules.glass.meta_data import MetadataExtractor