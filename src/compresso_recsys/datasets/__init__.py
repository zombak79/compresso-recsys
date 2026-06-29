from .amazon2023 import AmazonReviews2023
from .base import RecSysDataset, SplitBundle
from .movielens20m import MovieLens20M
from .movielens1m import MovieLens1M
from .goodbooks import Goodbooks

__all__ = [
    "SplitBundle",
    "RecSysDataset",
    "MovieLens1M",
    "MovieLens20M",
    "Goodbooks",
    "AmazonReviews2023",
]
