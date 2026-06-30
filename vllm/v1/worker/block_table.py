# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class BlockTable:

    def __init__(
        self,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        pin_memory: bool,
        device: torch.device,
        num_layers: int,
        block_repr: torch.Tensor,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.pin_memory = pin_memory
        self.device = device

        self.block_table = torch.zeros(
            (max_num_reqs, max_num_blocks_per_req),
            device=self.device,
            dtype=torch.int32,
        )
        self.block_table.fill_(-1)
        self.block_table_cpu = torch.zeros(
            (max_num_reqs, max_num_blocks_per_req),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )

        # block_repr_table size 可能会很大, 通过限制 max_num_reqs 限制其大小
        self.block_repr_table = torch.zeros((num_layers, max_num_reqs, max_num_blocks_per_req) + block_repr.shape, dtype=block_repr.dtype, device=self.device)
        self.block_table_cpu.fill_(-1)
        self.block_table_np = self.block_table_cpu.numpy()
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        if not block_ids:
            return
        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table_np[row_idx, start:start + num_blocks] = block_ids

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        self.block_table_np[tgt, :num_blocks] = self.block_table_np[
            src, :num_blocks]
        # copy block repr from src row to tgt row for all layers
        self.block_repr_table[:, tgt, :num_blocks] = self.block_repr_table[:, src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks

    def swap_row(self, src: int, tgt: int) -> None:
        num_blocks_src = self.num_blocks_per_row[src]
        num_blocks_tgt = self.num_blocks_per_row[tgt]
        self.num_blocks_per_row[src] = num_blocks_tgt
        self.num_blocks_per_row[tgt] = num_blocks_src

        self.block_table_np[[src, tgt]] = self.block_table_np[[tgt, src]]

        # swap block repr
        self.block_repr_table[:, [src, tgt]] = self.block_repr_table[:, [src, tgt]]

    def commit(self, num_reqs: int) -> None:
        self.block_table[:num_reqs].copy_(self.block_table_cpu[:num_reqs],
                                          non_blocking=True)

    def clear(self) -> None:
        self.block_table.fill_(-1)
        self.block_table_cpu.fill_(-1)
        self.block_repr_table.fill_(-1)

    def get_device_tensor(self) -> torch.Tensor:
        """Ruturns the device tensor of the block table."""
        return self.block_table

    def get_cpu_tensor(self) -> torch.Tensor:
        """Returns the CPU tensor of the block table."""
        return self.block_table_cpu

    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        return self.block_table_np

    def get_block_repr_table(self) -> torch.Tensor:
        """Ruturns the device tensor of the block representative tokens"""
        return self.block_repr_table
