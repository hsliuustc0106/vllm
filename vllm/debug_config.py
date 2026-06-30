import os

class DebugConfig:
    def __init__(self):
        self.step_time = True
        self.step_batch = True
        self.management_breakdown = False
        self.hit_log = False
        self.miss_rate = False
        self.test = False
        self.swap_copy_ops = False

        # 跳过换入，会导致结果完全错误!
        self.bypass_swap_in = False

        self.offline_start_profile_step = 35
        self.offline_end_profile_step = 38

        assert self.offline_start_profile_step < self.offline_end_profile_step, "offline start_profile_step must be less than end_profile_step"

        if os.environ.get("VLLM_TORCH_PROFILER_DIR") is not None:
            self.profile = True
        else:
            self.profile = False

global_debug_config = DebugConfig()

