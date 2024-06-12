import abc
from typing import Dict


class DocObj(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def from_dic(d: Dict):
        pass
    @abc.abstractmethod
    def to_dic(self) -> Dict:
        pass


class ObjWithID(DocObj, abc.ABC):
    def __init__(self, oid: str):
        self.oid = oid
