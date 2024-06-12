import re
import traceback
from typing import Optional, Dict, List, Tuple

import numpy as np

from kieta_modules import Module

from kieta_data_objs import Document, Area

import Levenshtein

class FunctionalRecognizer(Module):
    _MODULE_TYPE = 'FunctionalRecognizer'
    def __init__(self, stage: int, parameters: Optional[Dict] = {}, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)

        self.common_prefixes = self.parameters.get("common_prefixes",'No\.;Ex\.;Number;Example').split(";")

        self.enforce_ordererd_numbering = self.parameters.get("enforce_ordererd_numbering", False)

        self.no_predefined_column_labels = self.parameters.get("no_predefined_column_labels", 1)-1
        self.no_predefined_row_labels = self.parameters.get("no_predefined_row_labels", 1)-1

    def execute(self, inpt: Document) -> Document:
        l_tables = list(inpt.get_area_type('Table'))

        # map for stubs for each table: Dict[table_id, str]
        already_appeared = dict()

        for table in l_tables:
            # default
            if 'cells' not in table.data.keys():
                continue

            # set first row as column label
            # for c in table.data['cells'][0]:
            #     if c:
            #         inpt.areas.byId[c].data['column_label'] = True

            already_appeared[table.oid] = set()

            # set first column as row label
            for ix, c in enumerate(table.data['cells']):
                if ix <= self.no_predefined_column_labels:
                    if c[0]:
                        already_appeared[table.oid].add(inpt.areas.byId[c[0]].data['content'])
                    else:
                        already_appeared[table.oid].add(None)

                # check if cell already appears in a stub --> then it is not a row label and the row is a column header
                if (not c[0] and None in already_appeared[table.oid]) or (c[0] and inpt.areas.byId[c[0]].data['content'] in already_appeared[table.oid]):
                    for cc in c:
                        if cc:
                            inpt.areas.byId[cc].data['column_label'] = True
                    continue
                else:
                    for ir, cc in enumerate(c):
                        if cc and ir <= self.no_predefined_row_labels:
                            inpt.areas.byId[cc].data['row_label'] = True
            
            if self.no_predefined_column_labels == 0 and self.no_predefined_row_labels == 0:
                inpt = self.get_numbered_rows(inpt, table)

        return inpt

    def get_numbered_rows(self, doc, table, threshold: int = 0.5) -> Tuple[Document, int]:
        # number with one of the prefixes
        rgx_prefix_and_number = re.compile(r"^(?:{})\s?(\d*(?:\.\d)?)$".format("|".join(self.common_prefixes))) 
        rgx_number_with_opt_del = re.compile(r'^(?<!\d)[+-]?(\d{1,3}(?:[.,]\d{3})*|[.,]?\d+)(?=[.,]\d{1,2})?[.,]?\d*(?!\d)')
        # regex_number = re.compile(r""'\d+(?=\.)?')
        def analyze_numbering(table_strings):
            # for ix, row in enumerate(table_strings):
            #     # print
            #     self.debug_msg(f"{ix}, {row}")
            
            rows = list()
            # rows first
            # for ix, row in enumerate(table.data['cells']):
            for ix, row in enumerate(table_strings):
                lengths = list()
                prev_val = list()
                prev_was_digit = False
                for iy, cell in enumerate(row):
                    if not cell:
                        continue
                    # digit = regex_number.match(doc.areas.byId[cell].data['content'])
                    # print(cell, regex_number.findall(cell.strip()))
                    digit = rgx_prefix_and_number.findall(cell.strip())
                    lengths.append(len(cell))
                    if digit and len(digit) > 0 and digit[0]:
                        prev_val.append([int(digit[0])])
                        prev_was_digit = True
                    else:
                        if len(cell) == 0 or prev_was_digit:
                            # prev_val.append(0)
                            continue
                        else:
                            # check if it's a string, convert to ascii and check if the distance is 1
                            # this only considers last character
                            try:
                                prev_val.append([ord(x) for x in cell.strip()])
                            except Exception as e:
                                self.error_msg(f"{e} {traceback.format_exc()}")
                
                # print([1 if prev_val[i+1] - prev_val[i] == 1 else 0 for i in range(len(prev_val)-1)], len(prev_val)*threshold )
                # check if more than half of the values are increasing by 1
                # if sum([1 for i in range(len(prev_val)-1) if prev_val[i+1] - prev_val[i] == 1]) > len(prev_val) / 2:
                # print(prev_val, [1 for i in range(len(prev_val)-1) if prev_val[i+1] - prev_val[i] == 1], (len(prev_val)-1)*threshold)
                try:
                    if len(prev_val) > 1 and \
                    sum([1 for i in range(len(prev_val)-1) if prev_val[i+1][-1] - prev_val[i][-1] == 1 and len(prev_val[i+1]) == len(prev_val[i])]) >= (len(prev_val)-1)*threshold:               
                        rows.append(ix)
                except IndexError as e:
                    pass
                    # self.error_msg(f"{e} {traceback.format_exc()}")
            return rows
        
        def analyze_starting_rows(rows):
            # to account for multiple header rows
            count = 0
            for row in rows:
                # Check if the row is not completely filled or contains data cells
                if not all(cell for cell in row):
                    count += 1
                else:
                    count += 1
                    break
            # majority vote
            for ix, row in enumerate(rows[:count]):
                ls = [re.search(rgx_number_with_opt_del, doc.get_area_obj(cell).data['content']) for cell in row if cell] 
                # count 
                y,n = 0,0
                for l in ls:
                    if l:
                        y += 1
                    else:
                        n += 1
                if y > n:
                    return list(range(ix - 1))
            return list(range(count))

        # check if there are table rows and columns containing cells with continously increasing numbers
        # e.g. 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        # or No. 1, No. 2, No. 3
        # or Example 1, Example 2, Example 3
        # or A B C D E F G

        table_strings_rows = [[doc.areas.byId[c].data['content'] if c else '' for c in row] for row in table.data['cells']]
        # transpose table
        transposed = np.transpose(table.data['cells']).tolist()
        table_strings_cols = [[doc.areas.byId[c].data['content'] if c else '' for c in col] for col in transposed]

        rows = analyze_numbering(table_strings_rows)
        cols = analyze_numbering(table_strings_cols)

        self.debug_msg(f'Found {len(rows)} rows with increasing numbers in table {table.oid}, {rows}')
        self.debug_msg(f'Found {len(cols)} columns with increasing numbers in table {table.oid}, {cols}')

        # has to be before analyzation of col headers
        if len(cols) != 0 and len(rows) == 0:  # it is a transposed table, save it as such for later
            table.data['transposed'] = True

        # starting from first row go down as long as the row is not completely filled, or there is a row containing data cells
        # separate analyzation of multi-level col labels
        rows += analyze_starting_rows(table.data['cells'])
        rows = set(rows)
        self.debug_msg(f'Found {len(rows)} rows with increasing numbers in table {table.oid}, {rows}')

        #  set these rows as column labels
        try:
            for row in rows:
                for cell in table.data['cells'][row]:
                    if cell:
                        doc.areas.byId[cell].data['column_label'] = True
        except Exception as e:
            print(rows, len(table.data['cells'][row]))
            self.error_msg(f"{e} {traceback.format_exc()}")
            quit()
        try:
            for col in cols:
                for row in table.data['cells']:
                    if row[col]:
                        doc.areas.byId[row[col]].data['row_label'] = True
                # for cell in table.data['cells'][col]:
                #     if cell:
                #         doc.areas.byId[cell].data['row_label'] = True
        except Exception as e:
            print(cols, len())
            self.error_msg(f"{e} {traceback.format_exc()}")
            quit()
        return doc
