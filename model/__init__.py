"""Model package for MGPAN."""

from model.attention import NodeTypeAwarePooling
from model.mgpan import MGPAN, MGPANGraph, MGPANLayer

__all__ = ["MGPAN", "MGPANGraph", "MGPANLayer", "NodeTypeAwarePooling"]