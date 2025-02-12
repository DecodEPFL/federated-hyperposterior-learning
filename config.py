import torch
import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# SYNTH_PV_DIR = os.path.join(os.path.dirname(BASE_DIR), 'Synthetic_PV_Profiles')
SYNTH_PV_DIR = os.path.join(BASE_DIR, 'third_party', 'Synthetic_PV_Profiles')
PVDATA_DIR = os.path.join(SYNTH_PV_DIR, 'saved_results')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

disable_gpu = True
if disable_gpu:
    device = torch.device('cpu')
else:
    core_num = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(core_num)
        device = torch.device('cuda:'+str(core_num))
    else:
        device = torch.device('cpu')


def init():
    sys.path.insert(1, BASE_DIR)
    sys.path.insert(1, os.path.join(BASE_DIR, 'utils'))
    sys.path.insert(1, SYNTH_PV_DIR)
