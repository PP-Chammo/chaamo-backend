from enum import IntEnum
from typing import Optional


class CategoryId(IntEnum):
    """Category IDs matching the categories table in Supabase"""

    TOPPS = 1
    PANINI = 2
    FUTERA = 3
    POKEMON = 4
    DC = 5
    FORTNITE = 6
    MARVEL = 7
    GARBAGE_PAIL_KIDS = 8
    DIGIMON = 9
    WRESTLING = 11
    YU_GI_OH = 12
    LORCANA = 13
    OTHER = 99

    @classmethod
    def get_name(cls, category_id: int) -> str:
        """Get display name for category ID"""
        name_map = {
            1: "Topps",
            2: "Panini",
            3: "Futera",
            4: "Pokemon",
            5: "DC",
            6: "Fortnite",
            7: "Marvel",
            8: "Garbage Pail Kids",
            9: "Digimon",
            11: "Wrestling",
            12: "Yu-Gi-Oh!",
            13: "Lorcana",
            99: "Other",
        }
        return name_map.get(category_id, "Unknown")

    @classmethod
    def from_optional(cls, value: Optional[int]) -> Optional["CategoryId"]:
        """Convert optional int to CategoryId, return None if invalid"""
        if value is None:
            return None
        try:
            return cls(value)
        except ValueError:
            return None
