from __future__ import annotations

import math
from typing import Dict, Tuple


def check_space(func):
    def wrapper(self, other):
        if self.img_sp != other.img_sp:
            raise ValueError(
                f"Bounding boxes are not in same space, this ({self}), other ({other})"
            )
        return func(self, other)

    return wrapper


class BoundingBox():
    def __init__(
        self,
        x1: float = 0,
        y1: float = 0,
        x2: float = 0,
        y2: float = 0,
        img_sp: bool = False,
    ):
        # assert 0<= x1 and 0 <= x2 and 0 <= y1 and 0 <= y2, f"Invalid bounding box: ({x1}, {y1}), ({x2}, {y2})"
        # if < 0, set to 0
        if x1 < 0:
            x1 = 0
        if x2 < 0:
            x2 = 0
        if y1 < 0:
            y1 = 0
        if y2 < 0:
            y2 = 0

        self.x1: float = x1 if x1 < x2 else x2
        self.y1: float = y1 if y1 < y2 else y2
        self.x2: float = x2 if x1 < x2 else x1
        self.y2: float = y2 if y1 < y2 else y1
        self._img_sp: bool = img_sp
    
    @property
    def img_sp(self):
        return self._img_sp
    
    def is_vertical(self) -> bool:
        return self.height > self.width
    
    def is_horizontal(self) -> bool:
        return self.width > self.height

    def get_in_img_space(
        self, width_factor: float, height_factor: float
    ) -> BoundingBox:
        if self.img_sp:
            return self
        else:
            return BoundingBox(
                x1=self.x1 * width_factor,
                y1=self.y1 * height_factor,
                x2=self.x2 * width_factor,
                y2=self.y2 * height_factor,
                img_sp=True,
            )

    def get_in_xml_space(
        self, width_factor: float, height_factor: float
    ) -> BoundingBox:
        if self.img_sp:
            return BoundingBox(
                x1=self.x1 / width_factor,
                y1=self.y1 / height_factor,
                x2=self.x2 / width_factor,
                y2=self.y2 / height_factor,
                img_sp=False,
            )
        else:
            return self

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    def __repr__(self) -> str:
        return f"<{'IMG' if self.img_sp else 'XML'}>[({self.x1}, {self.y1}), ({self.x2}, {self.y2})]"

    def __eq__(self, other: BoundingBox) -> bool:
        return (
            self.x1 == other.x1
            and self.y1 == other.y1
            and self.x2 == other.x2
            and self.y2 == other.y2
            and self.img_sp == other.img_sp
        )

    def __getitem__(self, item):
        if item == 0:
            return self.x1
        elif item == 1:
            return self.y1
        elif item == 2:
            return self.x2
        elif item == 3:
            return self.y2
        else:
            raise IndexError("Index out of range")
    
    def __contains__(self, bb: BoundingBox) -> bool:
        """
        Check if the bounding box is inside the other bounding box
        """
        return self.x1 <= bb.x1 and self.y1 <= bb.y1 and self.x2 >= bb.x2 and self.y2 >= bb.y2

    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @check_space
    def intersection(self, other: BoundingBox):
        return BoundingBox(
            max(self.x1, other.x1),
            max(self.y1, other.y1),
            min(self.x2, other.x2),
            min(self.y2, other.y2),
            self.img_sp,
        )
    
    @check_space
    def intersects(self, other: BoundingBox) -> bool:
        return (
            self.x1 < other.x2
            and self.x2 > other.x1
            and self.y1 < other.y2
            and self.y2 > other.y1
        )

    @check_space
    def union(self, other: BoundingBox):
        return BoundingBox(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2),
            self.img_sp,
        )

    def middle(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5

    def __contains__(self, item):
        if isinstance(item, BoundingBox):
            return (
                self.x1 <= item.x1
                and self.x2 >= item.x2
                and self.y1 <= item.y1
                and self.y2 >= item.y2
            )
        else:
            return False

    def tuple(self):
        return self.x1, self.y1, self.x2, self.y2

    def polygon(self):
        return [
            (self.x1, self.y1),
            (self.x2, self.y1),
            (self.x2, self.y2),
            (self.x1, self.y2),
        ]

    @check_space
    def overlap(self, other: BoundingBox) -> bool:
        # Check if one rectangle is to the left of the other
        if self.x2 <= other.x1 or other.x2 <= self.x1:
            return False

        # Check if one rectangle is above the other
        if self.y2 <= other.y1 or other.y2 <= self.y1:
            return False

        # If neither of the above conditions are met, the rectangles overlap
        return True

    @check_space
    def intersection_over_union(self, other: BoundingBox) -> float:
        # Calculate the coordinates of the intersection rectangle
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        # If the intersection is negative, return 0
        if x2 <= x1 or y2 <= y1:
            return 0

        # Calculate the area of intersection rectangle
        intersection_area = (x2 - x1) * (y2 - y1)

        # Calculate the area of both bounding boxes
        box1_area = (self.x2 - self.x1) * (self.y2 - self.y1)
        box2_area = (other.x2 - other.x1) * (other.y2 - other.y1)

        # Calculate the union area
        union_area = box1_area + box2_area - intersection_area

        # Calculate the IoU
        iou = intersection_area / union_area

        return iou

    @check_space
    def overlap_horizontally(self, other: BoundingBox) -> bool:
        if self.x1 > other.x2 or self.x2 < other.x1:
            return False  # No overlap on x-axis
        return True  # Overlap detected

    @check_space
    def overlap_horizontally_right(self, other: BoundingBox) -> bool:
        return (
            True
            if range(max(int(self.y1), int(other.y1)), min(int(self.y2), int(other.y2)))
            and other.x1 > self.x1
            else False
        )

    @check_space
    def overlap_horizontally_left(self, other: BoundingBox) -> bool:
        return (
            True
            if range(max(int(self.y1), int(other.y1)), min(int(self.y2), int(other.y2)))
            and other.x1 < self.x1
            else False
        )

    @check_space
    def overlap_vertically(self, other: BoundingBox) -> bool:
        if self.y1 > other.y2 or self.y2 < other.y1:
            return False  # No overlap on y-axis
        return True  # Overlap detected

    @check_space
    def expand(self, other: BoundingBox) -> None:
        self.x1 = min(self.x1, other.x1)
        self.x2 = max(self.x2, other.x2)
        self.y1 = min(self.y1, other.y1)
        self.y2 = max(self.y2, other.y2)

    @check_space
    def __expand__(self, other: BoundingBox) -> BoundingBox:
        return BoundingBox(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2),
            self.img_sp,
        )

    @check_space
    def expand_average(self, other: BoundingBox) -> None:
        self.x1 = (self.x1 + other.x1) * 0.5
        self.x2 = (self.x2 + other.x2) * 0.5
        self.y1 = (self.y1 + other.y1) * 0.5
        self.y2 = (self.y2 + other.y2) * 0.5

    def expand_by(self, x1, y1, x2, y2) -> None:
        self.x1 = self.x1 + x1
        self.x2 = self.x2 + x2
        self.y1 = self.y1 + y1
        self.y2 = self.y2 + y2

    def reduce_by(self, x1, y1, x2, y2) -> None:
        self.x1 = self.x1 - x1
        self.x2 = self.x2 - x2
        self.y1 = self.y1 - y1
        self.y2 = self.y2 - y2

    @check_space
    def distance_horizontal(self, other: BoundingBox) -> float:
        left_edge_1, right_edge_1 = self.x1, self.x2
        left_edge_2, right_edge_2 = other.x1, other.x2

        if right_edge_1 < left_edge_2:
            distance = left_edge_2 - right_edge_1
        elif right_edge_2 < left_edge_1:
            distance = left_edge_1 - right_edge_2
        else:
            distance = 0

        return distance

    @check_space
    def distance_vertical(self, other: BoundingBox) -> float:
        top_edge_1, bottom_edge_1 = self.y1, self.y2
        top_edge_2, bottom_edge_2 = other.y1, other.y2

        if bottom_edge_1 < top_edge_2:
            distance = top_edge_2 - bottom_edge_1
        elif bottom_edge_2 < top_edge_1:
            distance = top_edge_1 - bottom_edge_2
        else:
            distance = 0

        return distance

    @check_space
    def distance(self, other: BoundingBox) -> float:
        x1, y1 = self.x1, self.y1
        x2, y2 = self.x2, self.y2
        x3, y3 = other.x1, other.y1
        x4, y4 = other.x2, other.y2

        # Calculate the shortest distance between two points on the lines
        numerator = abs((x4 - x3) * (y1 - y3) - (x1 - x3) * (y4 - y3))
        denominator = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if denominator == 0:
            # Lines are parallel or coincident
            return 0

        distance = numerator / denominator
        return distance

    def __transpose__(self):
        return BoundingBox(self.y1, -self.x2, self.y2, -self.x1, self.img_sp)

    def transpose(self):
        self.x1, self.y1, self.x2, self.y2 = self.y1, -self.x2, self.y2, -self.x1

    @staticmethod
    def from_dic(d: Dict):
        return BoundingBox(d["x1"], d["y1"], d["x2"], d["y2"], bool(d.get("img_sp", False)))

    def to_dic(self) -> Dict:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "img_sp": self.img_sp,
        }
