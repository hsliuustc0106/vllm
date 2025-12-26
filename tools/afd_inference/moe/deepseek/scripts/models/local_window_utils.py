import os
import torch
from torch.distributed.distributed_c10d import _world

window_size = 0
window_addr = None


def get_local_window():
    global window_addr, window_size
    if window_size == 0:
        raise RuntimeError(f"please alloc and exchange comm window first")
    return window_addr, window_size


def alloc_and_exchange_comm_window(peer_ranks, win_size, group=None):
    global window_addr, window_size
    if group is None:
        group = _world.default_pg
    group._get_backend(torch.device("npu"))._window_register_and_exchange(win_size, peer_ranks)
    window_tensor = group._get_backend(torch.device('npu'))._get_window_mem()
    window_addr = window_tensor.data_ptr()
    window_size = win_size
    print(f"window_register_and_exchange success, pid:{os.getpid()}, window addr={window_addr}, window_size={window_size}, peer_ranks:{peer_ranks}", flush=True)


def align_up(num, align: int = 512) -> int:
    return ((num + align - 1) // align) * align


def call_attn_win_size(micro_batch_num: int, micro_batch_size: int, selected_expert_num: int, hidden_size: int) -> int:
    '''
    计算attention侧 window 大小
    Args:
        micro_batch_num:
        micro_batch_size:
        selected_expert_num:
        hidden_size:
    Returns:
        attention侧 window 大小
    '''
    token_info_size = 4 * selected_expert_num * micro_batch_size * micro_batch_num
    # align to 512
    token_info_size = align_up(token_info_size, 512)
    token_data_size = 2 * hidden_size * selected_expert_num * micro_batch_size * micro_batch_num
    return token_info_size + token_data_size


def call_ffn_win_size(session_num: int, micro_batch_num: int, micro_batch_size: int, selected_expert_num: int,
                      hidden_size: int, quant_mode: int = 2) -> int:
    '''
    计算ffn侧 window 大小
    Args:
        session_num: attention worker num
        micro_batch_num:
        micro_batch_size:
        selected_expert_num:
        hidden_size:
        quant_mode: # 0: 非量化；1: 静态量化；2：动态量化
    Returns:
        ffn window size
    '''
    token_info_size = 4 * (selected_expert_num * micro_batch_size + 2) * micro_batch_num * session_num
    # align to 512
    token_info_size = align_up(token_info_size, 512)
    if quant_mode == 2:
        token_data_size = align_up(hidden_size + 4, 512)
    elif quant_mode == 0:
        token_data_size = hidden_size * 2
    else:
        token_data_size = hidden_size
    token_data_size = token_data_size * selected_expert_num * micro_batch_size * micro_batch_num * session_num
    return token_info_size + token_data_size
