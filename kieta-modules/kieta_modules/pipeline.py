from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple

import logging
import traceback
from tqdm import tqdm

import ast

from kieta_data_objs import Document
from . import Module


logger = logging.getLogger('main')


class ModuleConfigGenerator(ast.NodeVisitor):
    def __init__(self):
        self.modules = []

    def visit_ClassDef(self, node):
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == 'Module':
                self.process_module(node)

    def process_module(self, node):
        module_type = self.extract_module_type(node)
        if module_type is None:
            return

        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                params = self.process_parameters(item.args)

                module_info = {
                    "name": module_type,
                    "input": "Detection",
                    "output": "Detection",
                    "params": params
                }
                self.modules.append(module_info)

    def process_parameters(self, args):
        params = {}
        for arg in args.defaults:
            if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute) and arg.func.attr == 'get':
                param_name = arg.args[0].s
                default_value = arg.args[1].value
                params[param_name] = default_value
        return params

    def extract_module_type(self, node):
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and \
                            target.id == '_MODULE_TYPE' and \
                            isinstance(item.value, ast.Constant):
                        return item.value.value
        return None


class Pipeline:
    def __init__(self, name: str, pipeline: List[Dict[str, str]] = list()):
        self.name = name
        self.pipeline = []
        self.pipeline: List[Module] = [Module.create(k['name'], len(
            self.pipeline), k['parameter'] if 'parameter' in k.keys() else {}) for k in pipeline]

    def add_module(self, module: str, params, debug_mode: bool = False) -> None:
        try:
            self.pipeline.append(Module.create(
                module, len(self.pipeline), params, debug_mode))
        except Exception as e:
            logger.error(e, traceback.format_exc())

    def process(self, doc: Document) -> Iterable[Tuple[Document, str, int]]:
        if isinstance(doc, dict) and 'oid' in doc.keys():
            doc = Document.from_dic(doc)

        for ix, x in tqdm(enumerate(self.pipeline),
                          total=len(self.pipeline),
                          unit="modules"):
            try:
                if isinstance(doc, Document):
                    doc.add_revision(x._MODULE_TYPE)
                doc = x.execute(doc)

                yield (doc, x._MODULE_TYPE, 2)
            except Exception as e:
                logger.error(
                    f"Error in module {x._MODULE_TYPE} at step {ix}: {e} -->> {traceback.format_exc()}")
                yield (doc, x._MODULE_TYPE, 0)
                break

    def process_full(self, doc: Document) -> Document:
        if isinstance(doc, dict) and 'oid' in doc.keys():
            doc = Document.from_dic(doc)

        for ix, x in tqdm(enumerate(self.pipeline),
                          total=len(self.pipeline),
                          unit="modules",
                          desc=f"[Full]{doc.oid}" if hasattr(doc, 'oid') else "[Full]Document"):
            try:
                if isinstance(doc, Document):
                    doc.add_revision(x._MODULE_TYPE)

                doc = x.execute(doc)
                
            except Exception as e:
                logger.error(
                    f"Error in module {x._MODULE_TYPE} at step {ix}: {e} -->> {traceback.format_exc()}")
                break

        return doc
    
    def process_from_module(self, doc: Document, module_name: str = None, step_id: int = None) -> Document:
        if isinstance(doc, dict) and 'oid' in doc.keys():
            doc = Document.from_dic(doc)

        for ix, x in tqdm(enumerate(self.pipeline),
                          total=len(self.pipeline),
                          unit="modules",
                          desc=f"[Part]{doc.oid}" if hasattr(doc, 'oid') else "[Part]Document"):
            try:
                if module_name is not None and x._MODULE_TYPE == module_name:
                    step_id = ix
                if step_id is not None and ix == step_id:
                    if isinstance(doc, Document):
                        doc.add_revision(x._MODULE_TYPE)
                    doc = x.execute(doc)
            except Exception as e:
                logger.error(
                    f"Error in module {x._MODULE_TYPE} at step {ix}: {e} -->> {traceback.format_exc()}")
                break

        return doc

    def __len__(self):
        return len(self.pipeline)


class PipelineManager:
    def __init__(self, pipelines=dict()):
        self.pipelines: Dict[str, Pipeline] = pipelines
        self.celery_chains: Dict[str, Any] = None

    def get_pipeline(self, name: str) -> Pipeline:
        return self.pipelines.get(name, None)

    def add_pipeline(self, name: str, modules: List[Dict[str, str]] = list()):
        self.pipelines[name] = Pipeline(name, modules)

    def add_module(self, pipeline_name: str, module_name: str, params, debug_mode: bool = False):
        self.pipelines[pipeline_name].add_module(
            module=module_name, params=params, debug_mode=debug_mode)

    def clear(self) -> None:
        self.pipelines = dict()

    def read_from_file(self, path: str) -> None:
        import json
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                for pipeline in data:
                    self.add_pipeline(pipeline['name'], pipeline['modules'])
        except Exception as e:
            print(e)
            raise e
