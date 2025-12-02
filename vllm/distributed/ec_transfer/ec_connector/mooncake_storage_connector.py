# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import threading
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Optional, Union

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase, ECConnectorMetadata, ECConnectorRole)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

_EC_STORE_MODULES = {
    "datasystem":
    "vllm.distributed.ec_transfer.ec_lookup_buffer.datasystem_store",
    "mooncake": "vllm.distributed.ec_transfer.ec_lookup_buffer.mooncake_store"
}

ec_store_type = os.getenv("EC_STORE_TYPE", "mooncake")
async_handler = int(os.getenv("EC_STORE_ASYNC", 1))
module_name = _EC_STORE_MODULES.get(ec_store_type,
                                    _EC_STORE_MODULES["mooncake"])
ECMooncakeStore = import_module(module_name).ECMooncakeStore

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class MMMeta:
    req_id: str
    mm_hashes: set[str]

    @staticmethod
    def make_meta(req_id, mm_hashes) -> "MMMeta":
        return MMMeta(req_id=req_id, mm_hashes=mm_hashes)


@dataclass
class ECMooncakeStorageConnectorMetadata(ECConnectorMetadata):
    mm_datas: list[MMMeta]

    def __init__(self):
        self.mm_datas = []

    def add_mm_data(self, mm_data: MMMeta):
        self.mm_datas.append(mm_data)


class ECMooncakeStorageConnector(ECConnectorBase):

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        # req_id -> mm_hashes
        self._mm_datas_need_loads: dict[str, set[str]] = {}
        self.device = self._vllm_config.device_config.device
        if async_handler:
            self.loop = asyncio.new_event_loop()
            self.task_list = []
            self.task_list.append(self.loop.create_task(self._loader_loop()))
            self.task_list.append(self.loop.create_task(self._saver_loop()))

            self._pending_load_reqs: asyncio.Queue[tuple[str, list[str], dict[
                str, Any]]] = asyncio.Queue()
            self._pending_save_reqs: asyncio.Queue[tuple[str, list[str], dict[
                str, Any]]] = asyncio.Queue()

            self._finished_load_reqs: asyncio.Queue[str] = asyncio.Queue()
            self._finished_save_reqs: asyncio.Queue[str] = asyncio.Queue()

            thread = threading.Thread(target=self.start_event_loop,
                                      daemon=True)
            thread.start()

        self.store = ECMooncakeStore(vllm_config)

    def start_event_loop(self):
        """start event loop"""
        self.loop.run_until_complete(asyncio.gather(*self.task_list))
        self.loop.close()

    async def _loader_loop(self) -> None:
        while True:
            req_id, mm_hashes, encoder_cache = await self._pending_load_reqs.\
                get()
            tensors = self.store.batch_get(mm_hashes, self.device)
            for mm_hash, ec_cache in zip(mm_hashes, tensors):
                if ec_cache is None:
                    logger.error("Load failed for %s with error", mm_hash)
                encoder_cache[mm_hash] = ec_cache
            await self._finished_load_reqs.put(req_id)

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        """
        Start loading the cache from the connector into vLLM's encoder cache.

        This method loads the encoder cache based on metadata
        provided by the scheduler.It is called before `_gather_mm_embeddings`
        for the EC Connector. For EC,the `encoder_cache` and `mm_hash`
        are stored in `kwargs`.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping
                multimodal data hashes (`mm_hash`) to encoder cache tensors.
            kwargs (dict): Additional keyword arguments for the connector.
        """

        # Get the metadata
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, ECMooncakeStorageConnectorMetadata)
        assert encoder_cache is not None
        if not metadata.mm_datas:
            return

        for mm_data in metadata.mm_datas:
            mm_hashes = list(mm_data.mm_hashes)
            if async_handler:
                asyncio.run_coroutine_threadsafe(
                    self._pending_load_reqs.put(
                        (mm_data.req_id, mm_hashes, encoder_cache)), self.loop)
            else:
                tensors = self.store.batch_get(mm_hashes, self.device)
                for mm_hash, ec_cache in zip(mm_hashes, tensors):
                    encoder_cache[mm_hash] = ec_cache
                    if ec_cache is None:
                        logger.error("Load failed for %s", mm_hash)
                    logger.debug("Load tensor for %s successfully", mm_hash)

    async def _saver_loop(self):
        while True:
            req_id, mm_hashes, encoder_cache = await self._pending_save_reqs.\
                get()
            await self.store.batch_put(mm_hashes,
                                       [encoder_cache[h] for h in mm_hashes])
            await self._finished_save_reqs.put(req_id)

    def save_caches(self, encoder_cache, mm_hashes, **kwargs) -> None:
        """
        Save the encoder cache to the connector.

        This method saves the encoder cache from the worker's local storage
        to shared storage or another external connector.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping
                multimodal data hashes (`mm_hash`) to encoder cache tensors.
            mm_hashes (list[str]): The hash of the multimodal data whose cache
                is being saved.
            kwargs (dict): Additional keyword arguments for the connector.
        """
        if not self.is_producer:
            return
        assert encoder_cache is not None
        assert mm_hashes is not None
        req_id = kwargs.get("req_id")
        if req_id is None:
            raise ValueError("save caches requires a 'req_id' in kwargs")

        if not isinstance(req_id, str):
            raise TypeError(f"req_id must be a str, but got {type(req_id)}")
        if async_handler:
            asyncio.run_coroutine_threadsafe(
                self._pending_save_reqs.put(
                    (str(req_id), mm_hashes, encoder_cache)), self.loop)
        else:
            self.store.batch_put_async(mm_hashes,
                                       [encoder_cache[h] for h in mm_hashes])

    def wait_for_save(self):
        if async_handler:
            return
        else:
            self.store.wait_for_put()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if async_handler:
            finished_load = self._get_finished_queue_request(
                self._finished_load_reqs)
            finished_save = self._get_finished_queue_request(
                self._finished_save_reqs)
            return finished_save, finished_load
        return None, None

    def _get_finished_queue_request(self, q: asyncio.Queue) -> set[str]:
        finished_reqs = set()
        try:
            while True:
                req_id = q.get_nowait()
                finished_reqs.add(req_id)
        except asyncio.QueueEmpty:
            pass
        return finished_reqs

    def has_caches(
        self,
        request: "Request",
        index: Optional[int] = None,
    ) -> Union[tuple[Any, bool], Any]:
        """
        Check if cache exist externally for each mm_data of request

        Args:
            request (Request): the request object.
            index (Optional[int]) the index of mm data to check.

        Returns:
            List of bool indicate that ith mm_data exist in cache or not
        """
        if index is not None:
            res = self.store.batch_exists(
                [request.mm_features[index].identifier])[0]
        else:
            mm_hashes = [feature.identifier for feature in request.mm_features]
            res = self.store.batch_exists(mm_hashes)
        if async_handler and not self.is_producer:
            return res, True
        return res, False

    def update_state_after_alloc(
        self,
        request: "Request",
        index: int,
    ) -> None:
        """
        Update ECConnector state after encoder cache allocation.
        """
        mm_hash = request.mm_features[index].identifier
        # Insert mm_hash only if this block has not been recorded yet.
        self._mm_datas_need_loads.setdefault(request.request_id,
                                             set()).add(mm_hash)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ECConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.
        This only build for load mm_data only
        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = ECMooncakeStorageConnectorMetadata()
        for mm_hash, num_encoder_token in self._mm_datas_need_loads.items():
            meta.add_mm_data(MMMeta.make_meta(mm_hash, num_encoder_token))
        self._mm_datas_need_loads.clear()
        return meta

    def request_finished(
            self, request: "Request") -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its encoder cache is freed.

        Returns:
            True if the request is being saved/sent asynchronously and cached
            should not be freed until the request_id is returned from
            get_finished().
        """
        if async_handler and self.is_producer \
                and request.encoder_inputs_to_schedule:
            return True, None
        return False, None
