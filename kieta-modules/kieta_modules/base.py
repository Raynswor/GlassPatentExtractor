from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from kieta_data_objs import Area


logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)


class Module(ABC):
    subclasses = {}

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = False) -> None:
        self.stage = stage
        self.parameters = parameters if parameters else dict()


        self.apply_to: List[Area] = self.parameters.get('apply_to', [])

        self.debug_mode = debug_mode
        self.logger = logging.getLogger('main')
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls._MODULE_TYPE] = cls
    
    @classmethod
    def create(cls, module_type, stage, params, debug_mode: bool = False):
        if module_type not in cls.subclasses:
            raise ValueError('Bad module type {}'.format(module_type))

        return cls.subclasses[module_type](stage, parameters=params, debug_mode=debug_mode)

    @abstractmethod
    def execute(self, inpt: Any) -> Any:
        raise NotImplementedError

    def __str__(self) -> str:
        return self._MODULE_TYPE

    def debug_msg(self, msg: str) -> None:
        if self.debug_mode:
            with logging_redirect_tqdm(loggers=[self.logger]):
                self.logger.debug(msg)

    def error_msg(self, msg: str) -> None:
        with logging_redirect_tqdm(loggers=[self.logger]):
            self.logger.error(msg)

    def info_msg(self, msg: str) -> None:
        with logging_redirect_tqdm(loggers=[self.logger]):
            self.logger.info(msg)

    def warning_msg(self, msg: str) -> None:
        with logging_redirect_tqdm(loggers=[self.logger]):
            self.logger.warning(msg)
    
    def get_progress_bar(self, iterable, desc: str=None, unit: str=None, total: int = None):
        if self.debug_mode:
            return iterable
        else:
            desc = desc if desc else self._MODULE_TYPE
            total = total if total else len(iterable)
            return tqdm(iterable, desc=desc, unit=unit, total=total, leave=False)

    
