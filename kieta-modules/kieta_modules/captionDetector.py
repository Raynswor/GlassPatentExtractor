import collections
from copy import copy
import re
from typing import Dict, Iterable, List, Optional, Tuple

from kieta_data_objs import Document, BoundingBox
from kieta_modules import Module, util
import logging

logger = logging.getLogger('main')


def roman_to_arabic(roman_numeral):
    roman_numerals = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    
    total = 0
    prev_value = 0
    
    for numeral in reversed(roman_numeral):
        value = roman_numerals[numeral]
        
        if value < prev_value:
            total -= value
        else:
            total += value
        
        prev_value = value
    
    return total


class CaptionDetector(Module):
    _MODULE_TYPE = 'CaptionDetector'

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
        if isinstance(parameters["keywords"], list):
            self.keywords = parameters["keywords"]
        else:
            self.keywords = [x.strip() for x in parameters.get("keywords", "").split(',')]
        if isinstance(parameters["delimiters"], list):
            self.delimiters = parameters["delimiters"]
        else:
            self.delimiters = [x.strip() for x in parameters.get("delimiters", "").split(',')]
        # self.distance_line_above = int(parameters["distance_line_above"])
        self.caption_expansion_threshold = int(parameters.get("expansion_threshold", 5))
        self.removal = parameters.get("removal", [])
        self.keep_most_common = bool(parameters.get("keep_most_common", False))

        self.lines = parameters.get('lines', 'Line')
        self.strings = parameters.get('strings', 'String')

    
    def get_text_lines(self, doc: Document) -> Iterable[Tuple[str, List[str]]]:
        for l in doc.get_area_type(self.lines):
            # aggregate line
            line_string = list()
            line_objs = list()
            try:
                for xx in doc.references.byId[l.oid]:
                    try:
                        line_string.append(' '.join(doc.get_area_data_value(xx, 'content')))
                        line_objs.append(xx)
                    except KeyError:
                        pass
            except KeyError:
                pass
            yield ' '.join(line_string), line_objs

    def execute(self, inpt: Document) -> Document:
        """
        Detects table captions within document using pattern matching. Assumes that delimiter between identifier and caption string is the same for all tables
        """
        adds = list()

        for s, lines in self.get_text_lines(inpt):
            if (t := self.check_pattern(s))[0]:
                print(f"Found caption {s}")
                # search for next line below with threshold
                #get page
                page_id = f"page-{int(lines[0].split('-')[1])}"

                # todo: rotated tables are not working
                horizontal_lines, vertical_lines = util.sort_into_two_lists(inpt.get_area_type('DrawnLine', page_id), lambda x: x.is_horizontal())

                current = None
                content = s
                refs = list()
                for x in lines:
                    if current is None:
                        current = BoundingBox(inpt.areas.byId[x].boundingBox.x1, 
                                                inpt.areas.byId[x].boundingBox.y1,
                                                inpt.areas.byId[x].boundingBox.x2,
                                                inpt.areas.byId[x].boundingBox.y2,
                                                inpt.areas.byId[x].boundingBox.img_sp)
                    else:
                        current.expand(inpt.areas.byId[x].boundingBox)
                    try:
                        refs.extend(inpt.references.byId[x])
                    except KeyError:
                        pass
                    refs.append(x)

                stats = list()
                for other in list(inpt.get_area_type(self.lines, page_id)) + list(inpt.get_area_type(self.strings, page_id)):
                    # find below objects
                    if current.img_sp:
                        other_bb = other.boundingBox.get_in_img_space(inpt.pages.byId[page_id].factor_width, inpt.pages.byId[page_id].factor_height)
                    else:
                        other_bb = other.boundingBox.get_in_xml_space(inpt.pages.byId[page_id].factor_width, inpt.pages.byId[page_id].factor_height)
                    # print(other, other_bb, current)

                    other_content = ''.join(inpt.get_area_data_value(other, 'content'))

                    stats.append(
                    (other_content, content, other_bb.y1 - current.y2, other_bb.overlap_horizontally(current))
                    )
                    
                    if (abs(current.y1 - other_bb.y2) < self.caption_expansion_threshold or \
                    abs(other_bb.y1 - current.y2) < self.caption_expansion_threshold ) and \
                            other_bb.overlap_horizontally(current):

                        # ensure that there is no drawn line between current and other
                        # if there is, do not expand
                        # if not, expand
                        mock_expand = copy(current)
                        mock_expand.expand(other_bb)
                        found_intersect = False
                        for l in horizontal_lines:
                            if current.img_sp:
                                bounding = l.boundingBox.get_in_img_space(inpt.pages.byId[page_id].factor_width, inpt.pages.byId[page_id].factor_height)
                            else:
                                bounding = l.boundingBox.get_in_xml_space(inpt.pages.byId[page_id].factor_width, inpt.pages.byId[page_id].factor_height)
                            if bounding.intersects(mock_expand):
                                found_intersect = True
                        if found_intersect:
                            continue

                        current.expand(other_bb)
                        try:
                            refs.extend(inpt.references.byId[other])
                        except KeyError:
                            pass
                        refs.append(other)
                        try:
                            content += inpt.areas.byId[other].data['content']
                        except KeyError:
                            pass
                
                # for x in list(sorted(stats, key=lambda x: x[2])):
                #     print(x)
                # if current == lines:
                #     print(f"couldn't expand {s}")
                adds.append((page_id, current, content, t[1], refs))

                # check that distance to next line above is above threshold
                # other_lines = list()
                # for other in inpt.references.byId[f'page-{page}']:
                #     if inpt.areas.byId[other].boundingBox.overlap_vertically(current) and \
                #             current.y1 > inpt.areas.byId[other].boundingBox.y2:
                #         other_lines.append((inpt.areas.byId[other].boundingBox, current.y1 - inpt.areas.byId[other].boundingBox.y2 ))
                #
                # if min(other_lines, key=lambda x: x[1])[1] > self.distance_line_above:
                #     adds.append((f'page-{page}', current, content))


        # count combinations of (delimiter, keyword)
        # keep only most common
        captions_stats = collections.Counter([a[3] for a in adds])
        logger.debug(f"{inpt.oid} : Found {captions_stats} captions")
        try:
            most_common = captions_stats.most_common(1)[0][0]
        except IndexError:
            most_common = -1
        ix = 0

        for a in adds:
            if self.keep_most_common and a[3] != most_common:
                continue
            matches = re.findall(r"\b(?:\d+|[IVX]+)\b", a[2])
            
            ix += 1

            if len(matches) > 0:
                table_no = matches[0]
                # convert roman numerals to arabic
                if table_no.isupper():
                    self.debug_msg(f"Found roman numeral {table_no} converting to arabic")
                    table_no = roman_to_arabic(table_no)
            else:
                table_no = ix
                logger.info('did not find table number, using table caption index')
            self.debug_msg(f'TABLE NUMBER {table_no}')


            if any([x.lower() in a[2].lower() for x in self.removal]):
                self.debug_msg(f"Found continued table {a[2]}")
                print(inpt.add_area(page=a[0], category='Caption', boundingBox=a[1], data={'content': a[2], 'number': table_no, 'continued': True}, references=a[4]))
            else:
                print(inpt.add_area(page=a[0], category='Caption', boundingBox=a[1], data={'content': a[2], 'number': table_no}, references=a[4]))
        self.debug_msg(f"Found {ix} captions")
        # TODO: captions cannot cross drawn lines

        return inpt

    def check_pattern(self, s: str) -> Tuple[bool, Tuple[str, int]]:
        """
        Checks, if string fits pattern in patternstorage, returns delimiter

        :param s: string to check
        :return: [Result of check, delimiter]
        """

        sp = s.split(' ')
        if len(sp) >= 4:
            sp = sp[:4]

        try:
            sp.remove('')
        except ValueError:
            pass

        sp = [x for x in sp if x not in self.removal and x != '']

        if len(sp) == 0:
            return False, ("", -1)

        # make every substring combination of sp[0]
        combinations = set([sp[0]] + [sp[0][0:x] for x in range(1, len(sp[0]))])
        kw_index = -1
        if not sp:
            return False, ("", -1)
        elif sp[0].isupper():  # todo: no idea if that's working
            try:
                return True, ("", self.keywords.index(sp[0]))
            except ValueError:
                pass
        
        if (t:=set(self.keywords).intersection(combinations)):
            # index of keyword
            try:
                kw_index = self.keywords.index(list(t)[0])
            except Exception:
                pass
        else:
            return False, ("", -1)

        for x in range(len(sp)):
            for r in self.removal:
                sp[x] = sp[x].replace(r, '')

        try:
            if not (sp[-1][0].isupper() or sp[-2][0].isupper()):
                return False, ("", -1)
        except IndexError:
            pass
        
        try:
            for x in sp[1:]:
                if x[-1].isdigit():
                    return True, ('\n',kw_index)

                for p in self.delimiters:
                    if x[-1] == p:
                        return True, (p, kw_index)
        except IndexError as e:
            logger.error(f"Error in check_table_pattern: {e}")
            quit()

        # make the same for only sp[0]
        if sp[0][-1] in self.delimiters:
            return True, (sp[0][-1], kw_index)

        return False, ("", -1)


# if __name__ == '__main__':
#     test = CaptionDetector(0, {'delimiters': [':','.','-'], 'keep_most_common': True, 'expansion_threshold': 5, 'keywords': ['Table', 'Exhibit', 'TABLE', 'Figure'], 'removal':['continued', 'cont.']})
#
#     import json
#     with open('/home/sebastiank/Downloads/US9145333.json', 'r') as f:
#         tt = json.load(f)
#     inp = Document.from_dic(tt)
#
#     test.execute(inp)