from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys

modules = os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)