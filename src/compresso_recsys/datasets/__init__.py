from .amazon2023 import AmazonReviews2023
from .base import RecSysDataset, SplitBundle
from .movielens20m import MovieLens20M
from .movielens1m import MovieLens1M
from .goodbooks import Goodbooks
from .steam import Steam
from .netflix import NetflixPrize
from .taste_profile import TasteProfile
from .gowalla import Gowalla
from .multimodal import DBbook, LastFM2K

__all__ = [
    "SplitBundle",
    "RecSysDataset",
    "MovieLens1M",
    "MovieLens20M",
    "Goodbooks",
    "AmazonReviews2023",
    "Steam",
    "NetflixPrize",
    "TasteProfile",
    "Gowalla",
    "DBbook",
    "LastFM2K",
]
