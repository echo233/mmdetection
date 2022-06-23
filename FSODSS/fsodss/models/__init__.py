from .cosine_sim_bbox_head import CosineSimBBoxHead

from .bbox_head import TriBBoxHead
from .convfc_bbox_head import TriConvFCBBoxHead
from .base_roi_extractor import TriBaseRoIExtractor
from .tri_single_level_roi_extractor import TriSingleRoIExtractor

__all__ = [
    'CosineSimBBoxHead','TriBaseRoIExtractor', 'TriSingleRoIExtractor', 'TriBBoxHead','TriConvFCBBoxHead'
]
