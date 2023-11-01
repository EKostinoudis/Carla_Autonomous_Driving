from pathlib import Path
import sys

pathfile = str(Path(__file__).resolve().parent)
sys.path.insert(0, pathfile)

# TODO: remove these???
from network.models.architectures.CIL_multiview.CIL_multiview import CIL_multiview
from network.models.architectures.CIL_multiview.CIL_multiview_rllib import CIL_multiview_actor_critic
from configs import g_conf, merge_with_yaml
from dataloaders import make_data_loader, make_data_loader2
