import enum
from typing import Dict


class TypographicLayout(enum.Enum):
    MIXED = 0
    ONE_COLUMN = 1
    TWO_COLUMN = 2
    THREE_COLUMN = 3


class PageLayout:
    def __init__(self,
                 rotation: float = 0,
                 typographic_columns: TypographicLayout = "",
                 division: float = 0
                 ):
        self.rotation: float = rotation
        self.typographic_columns: TypographicLayout = typographic_columns
        self.division: float = division

    def __repr__(self):
        return f"<{self.rotation}°, {self.typographic_columns.name} at {self.division}>"

    @staticmethod
    def from_dic(d: Dict):
        return PageLayout(d['rotation'], d['typographic_columns'], d['division'])

    def to_dic(self) -> Dict:
        return {
            'rotation': self.rotation,
            'typographic_columns': self.typographic_columns if isinstance(self.typographic_columns, str) else self.typographic_columns.name,
            'division': self.division
        }