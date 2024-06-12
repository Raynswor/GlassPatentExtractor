import itertools
from kieta_modules import Module, GuppyOCRModule, TesseractOCRModule, BaselineDetector
from kieta_modules.util import nms_merge_with_index

from kieta_data_objs import Document, BoundingBox, Area

from typing import Any, Dict, List, Tuple


def clean(text: str) -> str:
    return text.strip().replace('—','').replace('-','').replace("'",'').replace("’",'').lower()


class MergeOCRModule(Module):
    _MODULE_TYPE="MergeOCRModule"

    """
    Merges different ocr results
    """

    def __init__(self, stage: int, parameters: Dict | None = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.mode = parameters.get('mode', 'Page')

        self.existing_text = parameters.get('existing_text', 'String')
        self.merging_text = parameters.get('merging_text', 'OCRString')

        self.existing_lines = parameters.get('existing_lines', 'Line')
        self.merging_lines = parameters.get('merging_lines', 'OCRLine')
        
        self.prev_revisions_to_merge = parameters.get('prev_revisions_to_merge', 2)
    
    def execute(self, inpt: Document) -> Document:
        revisions: Dict[Dict[str, List[Area]]] = dict()
        all_ids_to_merge = list()
        for rev in inpt.revisions[-self.prev_revisions_to_merge-1:-1]:
            areas = [inpt.get_area_obj(x) for x in rev.objects]
            all_ids_to_merge.extend(rev.objects)
            pages = dict()
            for a in areas:
                if 'content' not in a.data.keys():
                    continue
                if (t:=a.oid.split('-')[0]) not in pages:
                    pages[t] = [a]
                else:
                    pages[t].append(a)
            revisions[rev.comment] = pages
        
        
        for page in self.get_progress_bar(inpt.pages.allIds, unit="pages"):
            new_lines: List[Area] = list()
            new_strings: List[Area] = list()

            for values in revisions.values():
                for a in values[page.split('-')[-1]]:
                    if a.category == self.merging_lines:
                        new_lines.append(a)
                    elif a.category == self.merging_text:
                        new_strings.append(a)

            # get existing areas
            existing_strings: List[Area] = list()
            for ref in inpt.references.byId[page]:
                if ref in all_ids_to_merge:
                    continue
                t = inpt.get_area_obj(ref)
                if t.category == self.existing_text:
                    # change to img space
                    t.boundingBox = t.boundingBox.get_in_img_space(inpt.pages.byId[page].factor_width, inpt.pages.byId[page].factor_height)
                    existing_strings.append(t)
            
            
            high_conf, replace_with = self.merge_areas(inpt, existing_strings+new_strings)

            for oid in high_conf:
                inpt.areas.byId[oid].confidence = 1.0

            for (old, new) in replace_with:
                dbg_st = ""
                for n in new:
                    n.confidence = 1.0
                    dbg_st += f"{n.data['content']};"
                    n.category = "String"
                self.debug_msg(f"replacing {inpt.get_area_obj(old)} with {dbg_st}")
                inpt.replace_area_multiple(old, new)
        return inpt

    def merge_areas(self, 
                    inpt: Document, 
                    strings: List[Area]
                    ) -> Tuple[List[str], List[Tuple[str, List[Area]]]]:
        
        merged, indicces = nms_merge_with_index([b.boundingBox for b in strings], 0.4)

        for i in range(len(indicces)):
            indicces[i] = (strings[i], [strings[x] for x in indicces[i]])
        
        print('test')
        quit()
        
        # check which new areas only overlap with a single existing area
        # if text is similar, set high confidence
        high_conf = list()
        for (k,v) in list(new_overlaps_with.items()):
            if len(v) == 1:
                if clean(k.data['content']) == clean(v[0].data['content']):
                    high_conf.append(v[0].oid)
                    new_overlaps_with.pop(k)
                    old_overlaps_with.pop(v[0])
        
        replace_with = list()
        # iterate through old, if contains whitespace and there are #no_ws +1 new areas, replace with new
        for (k,v) in list(old_overlaps_with.items()):
            st = clean(k.data['content'])
            if len(st.split(' ')) + 1 == len(v):
                replace_with.append((k.oid, v))
                for x in v:
                    try:
                        new_overlaps_with.pop(x)
                    except KeyError:
                        pass
                old_overlaps_with.pop(k)
                
        self.info_msg(f"found {len(high_conf)} high confidence matches, {len(new_overlaps_with)} remaining")
        # debug print
        with open(f'{inpt.oid}_new_to_old.txt', 'a') as f:
            f.write('page\n')
            for (k,v) in sorted(new_overlaps_with.items(), key=lambda x: x[0].data['content']):
                content = ';'.join([f"{x.data['content']}" for x in v])
                f.write(f"{k.data['content']} -> {content}\n")
            f.write("\n\n\n")
        
        with open(f'{inpt.oid}_old_to_new.txt', 'a') as f:
            f.write('page\n')
            for (k,v) in sorted(old_overlaps_with.items(), key=lambda x: x[0].data['content']):
                content = ';'.join([f"{x.data['content']}" for x in v])
                f.write(f"{k.data['content']} -> {content}\n")
            f.write("\n\n\n")
            
        return high_conf, replace_with
        


