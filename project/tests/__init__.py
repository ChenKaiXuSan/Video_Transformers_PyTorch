try:
    from models import * 
except:
    import sys
    sys.path.append('/workspace/Walk_Video_PyTorch/project')
    from models import *