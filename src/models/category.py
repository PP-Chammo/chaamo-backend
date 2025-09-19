from enum import IntEnum, Enum
from typing import Optional


class CategoryId(IntEnum):
    """Category IDs matching the categories table in Supabase"""

    def __str__(self):
        # Show both ID and name for FastAPI docs dropdowns
        return f"{self.value} - {self.get_name(self.value)}"

    def __repr__(self):
        return f"{self.value} ({self.get_name(self.value)})"

    TOPPS = 1
    PANINI = 2
    FUTERA = 3
    POKEMON = 4
    DC = 5
    FORTNITE = 6
    MARVEL = 7
    GARBAGE_PAIL_KIDS = 8
    DIGIMON = 9
    WRESTLING = 10
    YU_GI_OH = 11
    LORCANA = 12
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
            10: "Wrestling",
            11: "Yu-Gi-Oh!",
            12: "Lorcana",
            99: "Other",
        }
        return name_map.get(category_id, "Unknown")

    @classmethod
    def get_keyword(cls, category_id: int) -> list[str]:
        """Get keyword for category ID"""
        name_map = {
            1: ["Topps"],
            2: ["Panini"],
            3: ["Futera"],
            4: ["Pokemon", "Pokémon"],
            5: ["DC"],
            6: ["Fortnite"],
            7: ["Marvel"],
            8: ["Garbage Pail Kids", "Pail Kids"],
            9: ["Digimon"],
            10: ["Wrestling"],
            11: ["Yu-Gi-Oh", "Yu~Gi~Oh"],
            12: ["Lorcana"],
            99: ["Other"],
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


class CategoryDropdown(str, Enum):
    TOPPS = "1 - Topps"
    PANINI = "2 - Panini"
    FUTERA = "3 - Futera"
    POKEMON = "4 - Pokemon"
    DC = "5 - DC"
    FORTNITE = "6 - Fortnite"
    MARVEL = "7 - Marvel"
    GARBAGE_PAIL_KIDS = "8 - Garbage Pail Kids"
    DIGIMON = "9 - Digimon"
    WRESTLING = "10 - Wrestling"
    YU_GI_OH = "11 - Yu-Gi-Oh!"
    LORCANA = "12 - Lorcana"
    OTHER = "99 - Other"

    @classmethod
    def to_category_id(cls, value: str) -> CategoryId:
        return CategoryId(int(value.split(" - ")[0]))
