"""
Types used throughout
"""

from __future__ import annotations

from typing import TypeVar

DataContainer = TypeVar("DataContainer")
DataContainer_co = TypeVar("DataContainer_co", covariant=True)
DataContainer_contra = TypeVar("DataContainer_contra", contravariant=True)
