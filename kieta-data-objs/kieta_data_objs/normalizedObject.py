from __future__ import annotations

from typing import TypeVar, List, Dict, Generic

from . import Page, Area, Link


T = TypeVar('T', Page, Area, List[str], Link)


class NormalizedObj(Generic[T]):
    def __init__(self, byId: Dict[str, T] = None):
        self.byId: Dict[str, T] = byId if byId else dict()
        self.allIds: List[str] = self.byId.keys()
    
    def __getitem__(self, key: str) -> T | None:
        return self.byId.get(key)

    def __len__(self):
        return len(self.byId)
    
    def __iter__(self):
        return iter(self.byId.values())
    
    def append(self, obj: T):
        self.byId[obj.oid] = obj
        # self.allIds.append(obj.oid)
    
    def remove(self, obj: T):
        try:
            if not isinstance(obj, str):
                del self.byId[obj.oid]
            else:
                del self.byId[obj]
        except KeyError:
            pass
        # self.allIds.remove(obj.oid)
    
    def remove_multiple(self, objs: List[T]):
        for obj in objs:
            self.remove(obj)

    @staticmethod
    def from_dic(dic, what: str) -> NormalizedObj:
        def rec_obj_creation(part, cls):
            d = {}
            for x in part.items():
                d[x[0]] = cls.from_dic(x[1])
            return d

        byId = None

        match what:
            case 'areas':
                byId = rec_obj_creation(dic['byId'], Area)
            case 'pages':
                byId = rec_obj_creation(dic['byId'], Page)
            case 'links':
                byId = rec_obj_creation(dic['byId'], Link)
            case 'references':
                byId = dic['byId']

        return NormalizedObj(byId) #, dic['allIds'])

    def to_dic(self) -> dict:
        # call to_dic() on each object in byId
        def rec_obj_to_dic(part):
            d = {}
            for x in part.items():
                try:
                    d[x[0]] = x[1].to_dic()
                except AttributeError:
                    d[x[0]] = x[1]
            return d
        
        return {
            'byId': rec_obj_to_dic(self.byId),
            'allIds': list(self.allIds)
        }