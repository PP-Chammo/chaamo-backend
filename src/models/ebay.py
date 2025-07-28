from enum import Enum


base_target_url = {
    "us": "https://www.ebay.com",
    "uk": "https://www.ebay.co.uk"
}

class Region(str, Enum):
    us = "us"
    uk = "uk"
