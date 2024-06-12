from kieta_modules import Module, util

from kieta_data_objs import Document, Area, BoundingBox, Page, util as data_util

from typing import Optional, Dict


class UtilityModule(Module):
    _MODULE_TYPE = 'UtilityModule'

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
    
    def execute(self, inpt: Document) -> Document:
        raise NotImplementedError("UtilityModule is an abstract class and cannot be executed")


class DeleteUtilityModule(UtilityModule):
    _MODULE_TYPE = 'DeleteUtilityModule'

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.delete = set(parameters.get('delete', ['String']))
        # none - all revision
        # int - last n revisions
        self.frm = parameters.get('from', None) 
    
    def execute(self, inpt: Document) -> Document:
        if self.frm is None:
            revisions = inpt.revisions
        else:
            revisions = inpt.revisions[-self.frm:]
        for rev in revisions:
            to_delete = []
            for oid in rev.objects:
                a = inpt.get_area_obj(oid)
                if not a:
                    to_delete.append(oid)
                    continue
                if a.category in self.delete:
                    inpt.delete_area(oid)
            for oid in to_delete:
                rev.objects.remove(oid)
        return inpt


class PageModifierUtilityModule(UtilityModule):
    _MODULE_TYPE = 'PageModifierUtilityModule'

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
        
    
    def execute(self, inpt: Document) -> Document:
        # half page pictures and make a new page from the other half
        new_pages = list()
        for p_id in inpt.pages.allIds:
            p = inpt.pages.byId[p_id]
            # half image
            half_bb = BoundingBox(0, 0, p.img_width//2, p.img_height, img_sp=True)
            other_half_bb = BoundingBox(p.img_width//2, 0, p.img_width-1, p.img_height, img_sp=True)


            new_number = inpt.pages.byId[p.oid].number+1
            new_page = Page(
                f"page-{new_number}", 
                number=new_number, 
                img=inpt.get_img_snippet_from_bb(other_half_bb, p.oid, True),
                img_width=p.img_width//2, img_height=p.img_height, 
                )
            

            inpt.pages.byId[p.oid].img = inpt.get_img_snippet_from_bb(half_bb, p.oid, True)

            data_util.base64_to_img(inpt.pages.byId[p.oid].img).show()

            inpt.pages.byId[p.oid].img_width = p.img_width//2

            new_pages.append(inpt.pages.byId[p.oid])
            new_pages.append(new_page)
        
        for idx, p in enumerate(new_pages):
            p.oid = f"page-{idx+1}"
            p.number = idx+1
            inpt.pages.byId[p.oid] = p
        
        inpt.pages.allIds = [p.oid for p in new_pages]
        inpt.references.allIds = [p.oid for p in new_pages]
        for p in new_pages:
            inpt.references.byId[p.oid] = []

        return inpt