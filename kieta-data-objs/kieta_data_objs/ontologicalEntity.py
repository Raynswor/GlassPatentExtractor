from typing import Dict, Union

from kieta_data_objs import DocObj, ObjWithID


class Entity(DocObj):
    def __init__(self, head: Union[str, ObjWithID],
                 relation: Union[str, ObjWithID],
                 tail: Union[str, ObjWithID], source: str = None, flags: Dict = None):
        self.source = source
        self.flags = flags
        self.head = head if isinstance(head, str) else head.oid
        if relation is not None:
            self.relation = relation if isinstance(relation, str) else relation.oid
        else:
            self.relation = None
        self.tail = tail if isinstance(tail, str) else tail.oid

    @staticmethod
    def from_dic(d: Dict):
        return Entity(d['head'], d.get('relation', None), d.get('tail', None), source=d.get('source', None), flags=d.get('flags', None))

    def __str__(self):
        return f"{self.head} - {self.relation} - {self.tail}"

#
# class OntologicalEntity(DocObj):
#     def __init__(self, subject: Union[str, ObjWithID], predicate: str, obj: Union[str, ObjWithID], source: str = None, flags: Dict = None):
#         self.flags = flags
#         self.source = source
#         self.subject = subject if isinstance(subject, str) else subject.oid
#         self.predicate = predicate
#         self.obj = obj if isinstance(object, str) else obj.oid
#
#     @staticmethod
#     def from_dic(d: Dict):
#         return OntologicalEntity(d['subject'], d['predicate'], d['object'], d['source'])