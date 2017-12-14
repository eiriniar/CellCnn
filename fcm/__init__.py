"""setup all things exported from FCM
"""

from core import FCMdata, FCMcollection
from core import Annotation
from core import PolyGate, points_in_poly, QuadGate, IntervalGate, ThresholdGate
from core import generate_f_score_gate
from core import BadFCMPointDataTypeError, UnimplementedFcsDataMode
from core import CompensationError
from core import load_compensate_matrix, compensate, gen_spill_matrix
from io import FCSreader, loadFCS, loadMultipleFCS, FlowjoWorkspace, load_flowjo_xml, export_fcs
from core import Subsample, SubsampleFactory, DropChannel
from core  import logicle, hyperlog

__all__ = [
            #Objects
            'FCMdata',
            'FCMcollection',
            'PolyGate',
            'QuadGate',
            'IntervalGate',
            'ThresholdGate',
            'FCSreader',
            'Annotation',
            'FlowjoWorkspace',
            #Exceptions
            'BadFCMPointDataTypeError',
            'UnimplementedFcsDataMode',
            'CompensationError',
            #functions
            'generate_f_score_gate',
            'logicle',
            'hyperlog',
            'loadFCS',
            'loadMultipleFCS',
            'load_compensate_matrix',
            'load_flowjo_xml',
            ]
