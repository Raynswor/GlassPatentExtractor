from calendar import month_name
import collections
from copy import copy
from dataclasses import dataclass
import traceback
from typing import Dict, List, Optional, Any, Set

import numpy as np

from dateutil.parser import parse

from owlready2 import *


from kieta_data_objs import Document, Area, BoundingBox

from kieta_modules import Module
from kieta_modules import util

import Levenshtein

import re


import logging
logger = logging.getLogger('main')

onto = get_ontology('http://uni-wuerzburg.de/glass_ontology.owl')

with onto:
    class Patent(Thing):
        pass

    class Glass(Thing):  # usually column
        pass

    class Attribute(Thing):  # data cell
        pass

    class Property(Thing):   # stub header
        pass

    class Unit(Thing):
        pass

    class has_title(Patent >> str, FunctionalProperty):
        pass

    class has_applicant(Patent >> str):
        pass

    class has_inventor(Patent >> str):
        pass

    class has_assignee(Patent >> str):
        pass

    class date_of_patent(Patent >> str, FunctionalProperty):
        pass

    class has_glass(Patent >> Glass):
        pass

    class has_abbrv(Property >> str):
        pass

    class has_value(Attribute >> str, FunctionalProperty):
        pass

    class has_unit(Attribute >> Unit, FunctionalProperty):
        pass   # for composition it's mol% wt% or mass%

    class has_attribute(Glass >> Attribute):
        pass

    # class has_synonyms(Unit >> str): pass

    mol = Unit("mol%", has_synonyms=[
               'mol %', 'molar percentage', 'mole percent', 'molar proportion'])
    wt = Unit("wt%", has_synonyms=['wt %', '% by weight', 'percentage by weight',
              '% by wt', 'percentage by wt', 'weight percent', 'weight %', 'weight'])
    mass = Unit("mass%", has_synonyms=[
                '% by mass', 'percentage by mass', 'mass %'])

    Unit.is_a.append(OneOf([mol, wt, mass]))


"""
10 NUMBER
45 Date of patent
54 Bezeichnung der Erfinung = title
71 Anmelder/Applicant
72 Erfinder/Inventor
73 Inhaber/Assignee 

75 = 71 + 72
76 = 71 + 72 + 73

Measure -> mol(_)% wt(_)% mass(_)%
"""


LINES = "Line"



def look_for_numbers(doc, areas: List[Area], numbers: List[int]) -> Dict[int, str]:
    """
    @return list of XMLString ids
    """
    numbers_in_parenthesis = {x: [f"({x})", f"[{x}]"] for x in numbers}
    # numbers = {x: [f"({x})"] for x in numbers}
    result = {}

    for a in areas:
        try:
            if a.category == LINES:
                content = ''.join([doc.areas.byId[x].data['content']
                                for x in doc.references.byId[a.oid]])
            else:
                content = a.data['content']
        except KeyError:
            continue
        content = content.replace(' ', '')
        # try:
        for n in numbers_in_parenthesis.items():
            for nn in n[1]:
                if nn in content:
                    try:
                        result[n[0]].add(a.oid)
                    except KeyError:
                        result[n[0]] = {a.oid}
                    # print('found', nn, 'in', content)
            # check if it's only number, no parenthesis
            if not n[0] in result.keys() and (content.startswith(str(n[0])) or content.startswith(f"({n[0]}") or content.startswith(f"[{n[0]}" or content.startswith(f"{n[0]})")) or content.startswith(f"{n[0]}]")):
                try:
                    result[n[0]].add(a.oid)
                except KeyError:
                    result[n[0]] = {a.oid}

    return result


def get_block_to_the_right(doc: Document,
                           page: str,
                           area_id: str,
                           two_column_layout: bool = True,
                           check_below: bool = False,
                           type: str = 'Paragraph'):
    paragraphs = list(doc.get_area_type(type, page))

    area = doc.get_area_obj(area_id)
    is_left = True

    # determine if area is in left or right column
    # find page middle over all pages
    page_middles = list()
    for p in doc.pages.allIds:
        page_middles.append(round(doc.pages.byId[p].img_width *
                            doc.pages.byId[p].layout.division, 2))
    page_middle = collections.Counter(page_middles).most_common(1)[0][0]
    # page middle is known here
    # page_middle = doc.pages.byId[page].layout.division
    if area.boundingBox.get_in_img_space(doc.pages.byId["page-0"].factor_width,
                                         doc.pages.byId["page-0"].factor_height).middle()[0] > page_middle:
        is_left = False
    try:
        print(f"String {area.data['content']} is in left column: {is_left}")
    except KeyError:
        print(f"Line {[doc.areas.byId[x].data['content'] for x in doc.references.byId[area_id]]} is in left column: {is_left}")

    if two_column_layout:
        if is_left:
            paragraphs = list(
                filter(lambda x: x.boundingBox.get_in_img_space(doc.pages.byId["page-0"].factor_width,
                                         doc.pages.byId["page-0"].factor_height).middle()[0] < page_middle, paragraphs))
        else:
            paragraphs = list(
                filter(lambda x: x.boundingBox.get_in_img_space(doc.pages.byId["page-0"].factor_width,
                                         doc.pages.byId["page-0"].factor_height).middle()[0] > page_middle, paragraphs))

    overlapping = list()
    img_bb = area.boundingBox.get_in_img_space(doc.pages.byId["page-0"].factor_width,
                                         doc.pages.byId["page-0"].factor_height)
    for a in paragraphs:
        # print(a.boundingBox, area.boundingBox, area.boundingBox.overlap_vertically(a.boundingBox)   )
        if img_bb.overlap_vertically(a.boundingBox.get_in_img_space(doc.pages.byId["page-0"].factor_width,
                                         doc.pages.byId["page-0"].factor_height)):
            overlapping.append(a)
    # overlapping, _ = util.sort_2D_grid(overlapping)

    if len(overlapping) == 0:
        return get_block_to_the_right(doc, page, area_id, two_column_layout, check_below, LINES)

    return util.sort_2D_grid(overlapping)[0]



def parse_glass_composition(doc: Document) -> List[owlready2.Thing]:
    entity_map: Dict[str, owlready2.Thing] = dict()

    counter = list()

    for x in onto.individuals():
        if type(x) == onto.Unit:
            entity_map[x.name] = x
            for l in x.has_synonyms:
                entity_map[l] = x

    for l in doc.get_area_type(LINES):
        line_string = ''.join(doc.get_area_data_value(l, "content")).lower().replace(' ', '').replace('(','').replace(')','')
        # line_string = l.lower()
        for k, v in entity_map.items():
            if k in line_string:
                counter.append(v)
                break

    return counter


def content(doc: Document, l: List[Area]) -> List[str]:
    return [doc.get_area_obj(rr).data['content'] for ll in l for rr in doc.references.byId[ll.oid]]



knowledge = {
    "SiO₂": [
        "SiO2",
        "SiO,",
        "Si0,",
        "Silicon oxide",
        "Sio2"
    ],
    "B₂O₃": [
        "B2O3",
        "B,O3",
        "B,03",
        "Boron oxide",
        "B2o3"
    ],
    "Li₂O": [
        "Li2O",
        "Li,O",
        "Li,0",
        "Lithium oxide",
        "Li2o"
    ],
    "MgO": [
        "MgO",
        "Mg0",
        "Mgo"
    ],
    "Na₂O": [
        "Na2O",
        "Na,O",
        "Na,0",
        "Sodium oxide"
    ],
    "Al₂O₃": [
        "Al2O3",
        "Al,O3",
        "Al,03",
        "AI203",
        "AI,03",
        "Aluminium oxide",
        "Al2o3"
    ],
    "Ga₂O₃": [
        "Ga2O3",
        "Ga,O3",
        "Ga,03",
        "Gallium oxide",
        "Ga2o3"
    ],
    "K₂O": [
        "K2O",
        "K20",
        "K,O",
        "K,0",
        "Κ20",
        "Potassium oxide",
        "K2o"
    ],
    "CaO": [
        "CaO",
        "Ca0",
        "Calcium oxide",
        "Cao"
    ],
    "BaO": [
        "BaO",
        "Ba0",
        "Barium oxide",
        "Bao"
    ],
    "SrO": [
        "SrO",
        "Sr0",
        "Strontium oxide",
        "Sro",
        "SRQ"
    ],
    "SnO₂": [
        "SnO2",
        "SnO,",
        "Sn0,",
        "Tin oxide",
        "Sn02",
        "Sno2"
    ],
    "ZrO₂": [
        "ZrO2",
        "ZrO,",
        "Zr0,",
        "2r0,",
        "Zirconium oxide",
        "Zro2"
    ],
    "TiO₂": [
        "TiO2",
        "TiO,",
        "Ti0,",
        "Titanium oxide",
        "Tio2"
    ],
    "CeO₂": [
        "CeO2",
        "CeO,",
        "Ce0,",
        "Cerium oxide",
        "Ce02",
        "Ceo2"
    ],
    "NAOH": [
        "NaOH",
        "Na0H",
        "Sodium hydroxide"
    ],
    "P₂O₅": [
        "P2O5",
        "P,O5",
        "P,05",
        "Phosphorus oxide",
        "P2o5"
    ],
    "R₂O": [
        "R2O",
        "R,O",
        "R,0",
        "R2o"
    ],
    "Fe₂O₃": [
        "Fe2O3",
        "Fe,O3",
        "Fe,03",
        "Iron oxide",
        "Fe2o3"
    ],
    "NH4F": [
        "NH4F",
        "Ammonium fluoride"
    ],
    "SO₃": [
        "SO3",
        "Sulfur oxide",
        "So3",
        "So0,"
    ],
    "Anneal point": [
        "Anneal pt.",
        "Anneal pt",
        "Anneal point"
        "Anneal"
    ],
    "Strain point": [
        "Strain pt.",
        "Strain pt",
        "Strain point",
        "Strain"
    ],
    "Softening point": [
        "Softening pt.",
        "Softening pt",
        "Softening point",
    ],
    "Youngs modulus": [
        "Young's modulus",
        "Young's mod.",
        "Young's",
        "Youngs modulus",
        "Youngs mod.",
    ],
    "Poisson ratio": [
        "Poisson's ratio",
        "Poisson's",
        "Poisson ratio",
        "Poisson"
    ],
    "Refractive index": [
        "Refractive index",
        "Refractive",
        "Refractive idx."
    ],
    "Glass transition point": [
        "Glass transition pt.",
        "Glass transition pt",
        "Glass transition point",
        "Glass transition"
    ],
    "Thermal expansion coefficient": [
        "Thermal expansion coeff.",
        "Thermal expansion coeff",
        "Thermal expansion coefficient",
        "Thermal expansion",
        "Themmal expansion"
    ],
    "Specific gravity": [
        "Specific gravity",
        "Specific grav.",
    ],
    "Fracture toughness": [
        "Fracture toughness",
        "Fracture tough.",
    ],
}

knowledge_val_to_key = dict()
for k, v in knowledge.items():
    for vv in v:
        knowledge_val_to_key[vv] = k
    knowledge_val_to_key[k] = k

@dataclass
class PropertyData:
    name: str
    unit: str

    def __init__(self, name: str, unit: str) -> None:
        # check name against knowledge with levensthein distance

        if name not in knowledge:
            for k in knowledge_val_to_key:
                t_name = name.replace(' ', '').replace('(','').replace(')','')
                if k == t_name or len(t_name) > 5 and Levenshtein.distance(k, t_name) <= 3:
                    name = knowledge_val_to_key[k]
                    break
        self.name = name

        self.unit = unit


# TODO: If there are multi-level col labels, only one is recognized. It should be all of them

class GlassExtractor(Module):
    _MODULE_TYPE = 'GlassExtractor'

    def __init__(self, stage: int, parameters: Optional[Dict] = {}, debug_mode: bool = True) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.pattern = r'[\[{(]([^\]\[\}\{\)\(]+)[\]})]'
        self.lines = parameters.get('lines', 'Line')
        self.strings = parameters.get('strings', 'String')

    def execute(self, inpt: Document):
        resource_refs, comp_unit = self.parse_titlepage(inpt)
        comp_unit = comp_unit.name if comp_unit else None
        resource_refs = resource_refs['hasDataResourceReference']
        # print(resource_refs)
        # print(comp_unit)

        setpoints: Dict[str, Dict] = dict()
        setpoints_to_page_table: Dict[str, Set[str]] = dict()
        table_col_to_setpoint = dict()

        table_order = list()
        prev_prefix = None
        try:
            for table_area in inpt.get_area_type('Table'):
                if "cells" not in table_area.data.keys():
                    continue
                table = None

                # rotate table so that 
                # examples are left-right
                # properties are top-down
                if table_area.data.get('transposed', False):
                    table = np.transpose(table_area.data['cells']).tolist()
                    self.info_msg(f'Transposed table {table_area.oid}')
                    # switch row and column labels
                    for i in range(len(table)):
                        for j in range(len(table[i])):
                            if not table[i][j]:
                                continue
                            area_obj = inpt.get_area_obj(table[i][j])

                            if 'row_label' in area_obj.data.keys() and 'column_label' in area_obj.data.keys():
                                continue
                            elif 'row_label' in area_obj.data.keys():
                                area_obj.data['column_label'], area_obj.data['row_label'] = True, False
                            elif 'column_label' in area_obj.data.keys():
                                area_obj.data['row_label'], area_obj.data['column_label'] = True, False
                else: 
                    table = copy(table_area.data['cells'])

                prefix_names = {"example", "number", "glass"}

                prefix = ''
                # column-wise
                try:                    
                    # check if first cell in this row contains a string that is not a number
                    # if so, use it as prefix for all following numbers
                    # if not, leave blank
                    for i in range(len(table)):  # each row
                        for j in range(len(table[i])):  # each cell in row
                            area_obj = inpt.get_area_obj(table[i][j])
                            if not table[i][j] or not area_obj.data.get('column_label', None)  or 'content' not in area_obj.data.keys():
                                continue

                            if area_obj.data.get('column_label', None) and area_obj.data.get('row_label', None):
                                if area_obj.data['content'].lower().strip() in prefix_names:
                                    prefix = area_obj.data['content']
                                continue
                            elif area_obj.data.get('row_label', None):
                                continue
                            
                            if area_obj.data['content'].lower().strip() in prefix_names:
                                prefix = area_obj.data['content']
                                continue
                            identifier = re.search(r"(?: |^|.)\d+\ *(\.|\,)?\ *\d*", area_obj.data['content'])
                            if identifier:
                                identifier = identifier[0].removeprefix('.').strip()
                                if "Ex" in area_obj.data['content']:
                                    identifier = f"Ex-{identifier}"
                                elif "No" in area_obj.data['content']:
                                    identifier = f"No-{identifier}"
                            if not identifier:
                                # NOTE: try out
                                # try:
                                #     identifier = area_obj.data['content']
                                # except KeyError:
                                #     continue
                                # self.debug_msg(f"Could not find identifier in {area_obj.data['content']}, using itself")
                                continue
                            # NOTE: current ocr detect '-' before numbers, where there are none. these are removed here, but it's not a good solution
                            identifier = identifier.replace(' ','').removeprefix('-')
                            if prefix != '':
                                identifier = f"{prefix}-{identifier}"
                            elif prev_prefix:
                                identifier = f"{prev_prefix}-{identifier}"
                                
                            if identifier not in setpoints.keys():
                                setpoints[identifier] = dict()
                                setpoints_to_page_table[identifier] = set()

                            table_col_to_setpoint[table_area.oid + str(j)] = identifier
                            setpoints_to_page_table[identifier].add((inpt.find_page_of_area(table_area.oid).split('-')[-1], table_area.oid))
                    # self.debug_msg(f"Found setpoints {setpoints}")
                    # self.debug_msg(f"Found table_col_to_setpoint {table_col_to_setpoint}")
                    # self.debug_msg(f"Found setpoints_to_page_table {setpoints_to_page_table}")
                    if prefix:
                        prev_prefix = prefix
                except IndexError:
                    self.error_msg(
                        f"Error while parsing tables: {traceback.format_exc()}")
                    continue
                
                # row index to property name
                properties: Dict[int, PropertyData] = dict()

                # property --> Ex

                property_placeholder_index = 0
                for row in range(len(table)):  # each row
                    for cell in range(len(table[row])):  # each cell in row
                        area_obj = inpt.get_area_obj(table[row][cell])
                        if not table[row][cell] or (area_obj.data.get('column_label', None) and not area_obj.data.get('row_label', None)):
                            continue

                        cont = area_obj.data['content'].strip()
                        # self.debug_msg(cont)
                        if cont == '':
                            continue
                        
                        # currently cannot handle multilevel row labels
                        if 'row_label' in area_obj.data.keys():
                            found_unit = re.findall(self.pattern, cont)
                            if found_unit:
                                cont = cont.replace(found_unit[0], '').translate(str.maketrans({
                                    '(': '',
                                    ')': '',
                                    '[': '',
                                    ']': '',
                                    '{': '',
                                    '}': '',
                                    ' ': '',
                                }))
                                found_unit = found_unit[0].replace(' ','')
                            # NOTE: this is just an observation and probably does not work for all cases
                            elif "temperature" in cont or "temp." in cont.lower() or "temp" in cont.lower():
                                found_unit = "°C"
                            else:
                                found_unit = comp_unit
                            
                            if row in properties.keys():
                                print(f"WARNING: row has label already {properties[row]}, expanding to {properties[row].name + cont}")
                                properties[row] = PropertyData(properties[row].name + cont, found_unit if found_unit != comp_unit else properties[row].unit)
                            else:
                                properties[row] = PropertyData(cont, found_unit)
                        elif area_obj.data.get('column_label', None) and not area_obj.data.get('row_label', None):
                            continue
                        else:
                            current_point = table_col_to_setpoint.get(table_area.oid+str(cell), None)
                            if current_point is None:
                                continue
                            # if no property is found, use a placeholder to not lose the data
                            if not row in properties.keys():
                                properties[row] = PropertyData(f'Miss-{property_placeholder_index}', 'Miss-')
                                property_placeholder_index += 1
                                self.error_msg(f"No property for row {row} in table {table_area.oid}")

                            # NOTE: current ocr detect .. between numbers, where there are none. these are removed here, but it's not a good solution
                            cont = cont.replace(' ', '').replace('. .', '.').replace('..', '.')

                            # special case for %
                            if '%' in cont:
                                cont = cont.replace('%', '').replace('^{}','')
                                if not properties[row].unit or properties[row].unit == 'Miss-':
                                    properties[row].unit = '%' 
                                else:
                                    pass
                                    # TODO what did i do here?
                                    # self.error_msg(f"Unit mismatch: {properties[row].unit} vs {comp_unit}")
                            setpoints[current_point][properties[row].name] = {
                                'value': cont,
                                'unit': properties[row].unit
                            }

                table_order.append(table_area.oid)
        except Exception as e:
            self.error_msg(
                f"Error while parsing tables: {e} -- {traceback.format_exc()}")

        result = list()

        patentNo = resource_refs['patentNo']
        del resource_refs['patentNo']
        pat = r"\d+"
        for ix, (k, v) in enumerate(setpoints.items()):
            if not v:
                continue
            ident = k

            numberOfComponent = len([y for y, x in v.items() if x['value'] != "N.A." and x['value'] != "0.0" and x['value'] != '' and (
                x['unit'] == 'mol%' or x['unit'] == 'wt%' or x['unit'] == 'mass%')])

            result.append({
                'generatedObjectIdentifier': f"{patentNo}-{ident}",
                'NumberOfComponent': numberOfComponent,
                'hasTableNumber': [int(inpt.get_area_obj(x[1]).data.get('number')) for x in setpoints_to_page_table[ident]],
                'hasPageNumber': [int(x[0]) for x in setpoints_to_page_table[ident]],
                'setpoints': v
            })

        # calculate table and column count
        table_count, column_count = 0, []
        already_seen = set()
        for x in inpt.areas.byId.values():
            if x.category == "Table" and (x.data['number'] not in already_seen or x.data.get('continued', False)):
                table_count += 1
                if 'cells' not in x.data.keys() or not x.data['cells']:
                    column_count.append(-1)
                    continue
                column_count.append(len(x.data['cells'][0]))
                already_seen.add(x.data['number'])
        
        # see if there is already information in the metadata
        return {
            'metadata': {
                "hasDatetime": time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime()),
                "isCreatedBy": "SeKe",
                "hasDataResourceReference": patentNo,
                "hasDataResourceName": patentNo,
                "hasDataResourceDescription": "patent extraction",
                "dataResourceSerialization": "json",
                **resource_refs,
                "hasPageCount": len(inpt.pages.allIds),
                "hasTableCount": table_count,
                "hasColumnCount": column_count
            },
            'steps': result
        }

    def parse_titlepage(self, doc: Document):
        map_inidCodes_to_onto = {
            10: [onto.Patent],
            11: [onto.Patent],
            43: [onto.date_of_patent],
            45: [onto.date_of_patent],
            54: [onto.has_title],
            71: [onto.has_applicant],
            72: [onto.has_inventor],
            73: [onto.has_assignee],
            75: [onto.has_applicant, onto.has_inventor],
            76: [onto.has_applicant, onto.has_inventor, onto.has_assignee]
        }
        candidate_strings: Dict[int, str] = look_for_numbers(doc, doc.get_area_by(
            lambda x: x.category in [self.strings, self.lines], 'page-0'), list(map_inidCodes_to_onto.keys()), )

        # # print(candidate_strings)

        # # check if prediction is unambiguous
        for k in candidate_strings.keys():
            if len(candidate_strings[k]) > 1:
                contents = list()
                self.warning_msg(f"Found multiple candidates for {k}")
                # pick the one that is surrounded by parantheses
                for c in candidate_strings[k]:
                    try:
                        con = doc.get_area_obj(c).data['content']
                    except KeyError:
                        con = " ".join([doc.areas.byId[x].data['content'] for x in doc.references.byId[c]])
                    contents.append(con.replace(' ',''))
                    if (con.startswith('(') and con.endswith(')')) or (con.startswith('[') and con.endswith(']')) or (con.startswith('{') and con.endswith('}')):
                        candidate_strings[k] = c
                        break
                if len(candidate_strings[k]) > 1:
                    # pick shortest
                    candidate_strings[k] = sorted(
                        zip(contents, candidate_strings[k]), key=lambda x: len(x[0]))[0][1]
                self.debug_msg(f"Chose {candidate_strings[k]}")
            else:  
                candidate_strings[k] = list(candidate_strings[k])[0]

        ## PATENT NUMBER
        if not doc.metadata.get('patent_number', None) or doc.metadata.get('patent_number') == 'nan' or not re.match(r'[A-Z]{2}\s*\d*((\/?\d*)|((\s\d*)*|(,\d*)*))(\s*[A-Z]\d)', doc.metadata['patent_number']):
            try:
                right_blocks = get_block_to_the_right(
                    doc, 'page-0', candidate_strings[10])
            except (KeyError, AttributeError):
                try:
                    right_blocks = get_block_to_the_right(
                        doc, 'page-0', candidate_strings[11])
                except (KeyError, AttributeError):
                    # right_blocks = doc.get_area_type('Line', 'page-0')
                    right_blocks = []
            res = None
            for a in right_blocks:
                text = ' '.join([doc.get_area_obj(b).data['content'] for b in doc.references.byId[a.oid]])
                res = re.search(r'[A-Z]{2}\s*\d*((\/?\d*)|((\s\d*)*|(,\d*)*))(\s*[A-Z]\d)', text)
                if res:
                    res = res[0]
                    break
            if not res:
                for a in right_blocks:
                    text = ' '.join([doc.get_area_obj(b).data['content'] for b in doc.references.byId[a.oid]])
                    res = re.search(r'\d{1,3}(,\d{3})*', text)
                    if res:
                        res = res[0]
                        break
            if not res:
                self.error_msg("Could not find patent number")
                res = doc.oid
        else:
            res = doc.metadata['patent_number']
        
        patent = onto.Patent(res)
        self.debug_msg(f"found patent {patent.name}")
        
        ## DATE OF PATENT
        if not doc.metadata.get('publication_date', None) or doc.metadata.get('publication_date') == 'nan':
            try:
                right_blocks = get_block_to_the_right(
                    doc, 'page-0', candidate_strings[45])
            except (KeyError, AttributeError):
                try:
                    right_blocks = get_block_to_the_right(
                        doc, 'page-0', candidate_strings[43])
                except (KeyError, AttributeError):
                    right_blocks = []
            text = ' '.join([''.join(doc.get_area_data_value(x, "content")) for x in right_blocks])

            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                      'Sep', 'Oct', 'Nov', 'Dec'] + [s for s in month_name[1:]]

            regex = fr"({'|'.join(months)})\s*.?\s*(.)?\d+\s*,\s*\d+"
            res = re.search(regex, ''.join(text))
            # print(res)
            if res:
                text = res.group(0)
                try:
                    parsed = parse(text, fuzzy=True)
                    patent.date_of_patent = parsed.strftime('%Y-%m-%d')
                except Exception as e:
                    self.error_msg(f"Error while parsing date: {e}")
        else:
            patent.date_of_patent = doc.metadata['publication_date']
        self.debug_msg(f'found date {patent.date_of_patent}')


        ## TITLE
        if not doc.metadata.get('title', None) or doc.metadata.get('title') == 'nan':
            try:
                title = content(doc,get_block_to_the_right(
                    doc, 'page-0', candidate_strings[54], True))
            except (KeyError, AttributeError):
                title = None
        else:
            title = doc.metadata['title']
        
        ## APPLICANT, INVENTOR, ASSIGNEE
        applicant = doc.metadata.get('applicant', None)
        inventor = doc.metadata.get('inventors', None)
        assignee = doc.metadata.get('assignee', None)

        if not applicant or applicant == 'nan':
            try:
                applicant = content(doc,get_block_to_the_right(
                    doc, 'page-0', candidate_strings[76], True))
                inventor = applicant if not inventor else inventor
                assignee = applicant if not assignee else assignee
            except (KeyError, AttributeError):
                pass
        if not applicant or applicant == 'nan':
            try:
                applicant = content(doc,get_block_to_the_right(
                    doc, 'page-0', candidate_strings[75], True))
                inventor = applicant if not inventor else inventor
                assignee = content(doc,get_block_to_the_right(doc, 'page-0', candidate_strings[73], True)) if not assignee else assignee
            except (KeyError, AttributeError):
                pass
        if not applicant or applicant == 'nan':
            try:
                applicant = content(doc,get_block_to_the_right(
                    doc, 'page-0', candidate_strings[71], True))
                inventor = content(doc,get_block_to_the_right(
                    doc, 'page-0', candidate_strings[72], True)) if not inventor else inventor
                assignee = content(doc,get_block_to_the_right(
                    doc, 'page-0', candidate_strings[73], True)) if not assignee else assignee
            except (KeyError, AttributeError):
                pass
    
        ## Summary
        summary = doc.metadata.get('abstract', None)
        ## Codes
        ipc_codes = doc.metadata.get('ipc', None)

        # def clean(x):
        #     try:
        #         return ' '.join(x).replace("Inventors", '').replace("Assignees", '')\
        #             .replace("Applicants", "").replace("Inventor", '').replace("Assignee", '')\
        #             .replace("Applicant", "").replace(':', '').replace('  ', ' ').strip()
        #     except TypeError:
        #         return ''
            
        # print(f"Found applicant {applicant}")
        # print(f"Found inventor {inventor}")
        # print(f"Found assignee {assignee}")
        # return {
        #     "hasDataResourceReference": {
        #         "patentNo": patent.name,
        #         "hasPatentTitle": clean(title),
        #         "hasApplicant": clean(applicant),
        #         "hasInventor": clean(inventor),
        #         "hasAssignee": clean(assignee),
        #         "hasDataResourceDate": patent.date_of_patent
        #     }
        # }, unit


        ## UNITS
        unit_candidates = collections.Counter(parse_glass_composition(doc))
        try:
            unit = unit_candidates.most_common(1)[0][0]
        except IndexError:
            unit = None
            
        return {
            "hasDataResourceReference": {
                "patentNo": patent.name,
                "hasPatentTitle": title,
                "hasApplicant": applicant,
                "hasInventor": inventor,
                "hasAssignee": assignee,
                "hasDataResourceDate": patent.date_of_patent,
                "hasAbstract": summary,
                "hasIPC": ipc_codes
            }
        }, unit
