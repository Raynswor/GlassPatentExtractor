from copy import copy
from typing import Optional, Dict, List

import numpy as np
import cv2

from kieta_data_objs import Document, BoundingBox, TypographicLayout
from kieta_data_objs.util import base64_to_img

from kieta_modules import Module
from kieta_modules.util import nms_merge, get_overlapping_areas


class ParagraphDetector(Module):
    _MODULE_TYPE = 'ParagraphDetector'

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)


        self.lines = parameters.get('lines', 'Line')
        self.strings = parameters.get('strings', 'String')

    def execute(self, inpt: Document) -> Document:
        layouts = list()
        minima = list()
        for page in inpt.pages.byId.values():
            image = np.array(base64_to_img(page.img))

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # blur = cv2.GaussianBlur(gray, (7, 7), 0)
            blur = cv2.GaussianBlur(gray, (43, 7), 0)

            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # debug plot

            # Create rectangular structuring element and dilate
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilate = cv2.dilate(thresh, kernel, iterations=4)


            # Find contours and draw rectangle
            cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cnts = [cv2.boundingRect(c) for c in cnts]
            cnts: List[BoundingBox] = [BoundingBox(x,y,x+w,y+h) for x,y,w,h in cnts]

            # draw dilate with bounding boxes
            # for c in cnts:
            #     cv2.rectangle(dilate, (c.x1, c.y1), (c.x2, c.y2), (255, 0, 0), 2)

            # debug plot
            # plt.imshow(dilate)
            # plt.waitforbuttonpress()

            # non maximum suppression iterative
            prev = 0
            iteration = 0
            do_again = 0
            while len(cnts) != prev or do_again < 2:
                # print(f"iteration {iteration}")
                prev = len(cnts)
                cnts = nms_merge(cnts, 0.05)
                iteration += 1
                if len(cnts) == prev:
                    do_again += 1
                else:
                    do_again = 0
            # debug_plot_bb(inpt, page.oid, cnts)

            # ignore boxes with small area or small height
            for c in cnts:
                # if c.area() > 1000 and c.height > 50:
                    # convert bb to pdf coordinates
                c.img_sp = True
                bbb = copy([inpt.areas.byId[xx] for xx in inpt.references.byId[page.oid]])
                for bb in bbb:
                    bb.boundingBox = bb.boundingBox.get_in_img_space(page.factor_width, page.factor_height)
                areas = get_overlapping_areas(c, bbb, [self.strings])


                inpt.add_area(page.oid, "Paragraph", c, references=[x.oid for x in areas[self.strings]])

        #     # determine typographic layout of page
        #     # apply top/bottom margins
            thresh[:int(thresh.shape[0] * 0.1), :] = 0
            thresh[int(thresh.shape[0] * 0.9):, :] = 0
            # sum and divide by 255
            number_of_white_pixels = np.sum(thresh, axis=0) / 255
            # non-mimimum suppression
            for i in range(1, len(number_of_white_pixels) - 1):
                if number_of_white_pixels[i] < number_of_white_pixels[i - 1] and number_of_white_pixels[i] < number_of_white_pixels[i + 1]:
                    number_of_white_pixels[i] = 0

            # apply left/right margins
            number_of_white_pixels[:int(len(number_of_white_pixels) * 0.1)] = 0
            number_of_white_pixels[int(len(number_of_white_pixels) * 0.9):] = 0

            # iteratively apply threshold
            # search for connected components
            # if there are more than 2, apply threshold again
            # if there are 2, check if they are left and right of the middle
            # do this until there are 2 connected components left or maximum iterations reached
            max_iterations = 10
            groups = list()
            while True and max_iterations > 0:
                # apply threshold
                number_of_white_pixels[number_of_white_pixels < np.mean(number_of_white_pixels)] = 0
                # search for connected components
                prev_value = 0
                con_com = list()
                for i in range(len(number_of_white_pixels)):
                    if number_of_white_pixels[i] != 0 and prev_value == 0:
                        con_com.append([i])
                    elif number_of_white_pixels[i] != 0 and prev_value != 0:
                        con_com[-1].append(i)
                    prev_value = number_of_white_pixels[i]
                if len(con_com) == 2:
                    # check if they are left and right of the middle
                    if con_com[0][0] < len(number_of_white_pixels) / 2 and con_com[1][0] > len(number_of_white_pixels) / 2:
                        groups = con_com
                        break
                groups = con_com
                max_iterations -= 1
            # merge group less than X pixels apart
            while True:
                # merge groups
                for i in range(len(groups) - 1):
                    if groups[i + 1][0] - groups[i][-1] < 50:
                        groups[i] = groups[i] + groups[i + 1]
                        groups.pop(i + 1)
                        break
                else:
                    break
            # debug plot
            # plt.plot(number_of_white_pixels)
            # for group in groups:
            #     plt.axvline(x=group[0], color='r')
            #     plt.axvline(x=group[-1], color='r')
            # plt.show()

            # check if there is minimum at the middle
            if len(groups) < 2:
                layouts.append(TypographicLayout.ONE_COLUMN)
            elif len(groups) == 2:
                layouts.append(TypographicLayout.TWO_COLUMN)
            else:
                layouts.append(TypographicLayout.MIXED)
            minima.append(groups)

        # if there are some pages with one column and some with two column, set one column to mixed
        if TypographicLayout.ONE_COLUMN in layouts and TypographicLayout.TWO_COLUMN in layouts:
            for i in range(len(layouts)):
                if layouts[i] == TypographicLayout.ONE_COLUMN:
                    layouts[i] = TypographicLayout.MIXED

        # search through groups and calculate average middlepoint of clusters (if there are two)
        middle = list()
        for group in minima:
            if len(group) == 2:
                middle.append((group[0][-1] + group[1][0]) / 2)
            else:
                middle.append(group[0][0])

        # set typographic layout
        for ix, (p, l) in enumerate(zip(inpt.pages.byId.values(), layouts)):
            p.set_typographic_layout(l)
            p.set_division(middle[ix] / p.img_width)
            self.debug_msg(f"page {p.oid} has layout {l} with division {middle[ix] / p.img_width}")
        return inpt