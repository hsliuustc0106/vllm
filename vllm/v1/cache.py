from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Tuple, Optional, List, Any
import heapq
from collections import deque, defaultdict
import random

class AbstractCache(ABC):
    def __init__(self, capacity):
        self.capacity = capacity

    @abstractmethod
    def get(self, block_key, hot_score):
        pass

    @abstractmethod
    def pin_block(self, block_key):
        pass

    @abstractmethod
    def unpin_block(self, block_key):
        pass

class LRUWithHotCache(AbstractCache):
    def __init__(self, capacity, decay_factor=0.44, window_size=110):
        """
        基于热度分数 + Sliding Window 的 KV Cache
        """
        self.capacity = capacity
        self.cache = {}  # block_key -> HotCacheEntry
        self.reverse_mapping = {}  # slot_id -> block_key
        self.free_slots = set(range(capacity))
        self.decay_factor = decay_factor
        self.window_size = window_size

        self.heap = []  # min-heap: (hot_score, version, block_key, timer)
        self.global_version = 0
        self.timer = 0

    def add_timer(self):
        self.timer += 1

    def _log_scale_hot_score(self, hot_score):
        """log 缩放热度，处理 inf 和极端值"""
        if hot_score == float('inf'):
            hot_score = 1e6
        elif hot_score == float('-inf'):
            hot_score = -1e6
        sign = 1 if hot_score >= 0 else -1
        return sign * math.tanh(math.log1p(abs(hot_score)))

    def get(self, block_key, hot_score=0) -> Tuple[int, bool]:
        """获取 block, 更新热度 + 时间窗口"""
        scaled_hot = self._log_scale_hot_score(hot_score)

        # Cache hit
        if block_key in self.cache:
            entry = self.cache[block_key]
            # 时间权重
            time_weight = max(0, 1 - (self.timer - entry.timer) / self.window_size)
            entry.hot_score = entry.hot_score * self.decay_factor + scaled_hot * (1 - self.decay_factor) * time_weight

            self.global_version += 1
            entry.version = self.global_version
            entry.timer = self.timer

            heapq.heappush(self.heap, (entry.hot_score, entry.version, block_key, self.timer))
            return entry.slot_id, True

        # Cache miss
        if len(self.cache) >= self.capacity:
            self.cleanup_heap()  # 小trick 1
            self._evict_by_window_or_heap()

        slot_id = self._allocate_slot()
        self.global_version += 1
        entry = HotCacheEntry(slot_id, scaled_hot, self.global_version, is_pinned=False, timer=self.timer)

        self.cache[block_key] = entry
        self.reverse_mapping[slot_id] = block_key
        heapq.heappush(self.heap, (entry.hot_score, entry.version, block_key, self.timer))

        return slot_id, False

    def _evict_by_window_or_heap(self):
        """优先驱逐超出窗口的 block，否则驱逐最低热度"""
        # Step1: 超过窗口时间的 block
        expired_keys = [k for k, v in self.cache.items() if v.timer < self.timer - self.window_size and not v.is_pinned]
        if expired_keys:
            for k in expired_keys:
                entry = self.cache[k]
                self.free_slots.add(entry.slot_id)
                del self.reverse_mapping[entry.slot_id]
                del self.cache[k]
            return

        # Step2: 使用堆驱逐最低热度
        while self.heap:
            hot_score, version, block_key, timer = heapq.heappop(self.heap)
            if block_key not in self.cache:
                continue
            entry = self.cache[block_key]
            if entry.version != version or entry.is_pinned:
                continue
            
            self.free_slots.add(entry.slot_id)
            del self.reverse_mapping[entry.slot_id]
            del self.cache[block_key]
            return

        raise RuntimeError("无法驱逐：所有 block 都被固定")

    def _allocate_slot(self):
        if not self.free_slots:
            raise RuntimeError("No free slots available")
        return self.free_slots.pop()

    def pin_block(self, block_key) -> bool:
        if block_key not in self.cache:
            return False
        self.cache[block_key].is_pinned = True
        return True

    def unpin_block(self, block_key) -> bool:
        if block_key not in self.cache:
            return False
        self.cache[block_key].is_pinned = False
        return True

    def cleanup_heap(self):
        """清理堆中无效记录"""
        valid_entries = [(hs, v, k, t) for hs, v, k, t in self.heap if k in self.cache and self.cache[k].version == v]
        self.heap = valid_entries
        heapq.heapify(self.heap)
       

class LRUCache(AbstractCache):
    def __init__(self, capacity):
        """
        初始化高效的 LRU Cache, 支持固定 block 功能

        Args:
            capacity: Cache 的容量(slot 数量)
        """
        super().__init__(capacity)
        self.cache = OrderedDict()  # {(layer_id, cpu_block_id): (slot_id, timer, is_pinned)}
        self.reverse_mapping = {}  # {slot_id: (layer_id, cpu_block_id)}
        self.free_slots = set(range(capacity))  # 跟踪空闲的 slot
        self.timer = 0

    def add_timer(self):
        self.timer += 1

    def get(self, block_key, hot_score=0.0) -> Tuple[int, bool]:
        """
        获取一个 block, 如果不在 cache 中则加载并分配 slot

        Args:
            block_key: (layer_id, cpu_block_id)
            hot_score: 块热度值

        Returns:
            slot_id: 分配给该 block 的 slot ID
            hit: 是否命中 cache
        """

        # 如果 block 在 cache 中
        if block_key in self.cache:
            slot_id, _, is_pinned = self.cache[block_key]
            # 更新访问时间和位置(LRU)
            self.cache.move_to_end(block_key, last=False)  # 移到最前面, 表示最近使用
            self.cache[block_key] = (slot_id, self.timer, is_pinned)
            return slot_id, True

        # Cache miss, 需要加载 block

        # 如果 cache 已满, 需要驱逐
        if len(self.cache) >= self.capacity:
            self._evict()

        # 分配一个新的 slot
        slot_id = self._allocate_slot()

        # 将新的 block 加入 cache(默认不固定)
        self.cache[block_key] = (slot_id, self.timer, False)
        self.cache.move_to_end(block_key, last=False)  # 移到最前面
        self.reverse_mapping[slot_id] = block_key

        return slot_id, False

    def _evict(self):
        """
        驱逐最不常用的非固定 block
        """
        # 从后向前遍历 (最不常用的在后面)
        for block_key, (slot_id, timer, is_pinned) in reversed(self.cache.items()):
            # 跳过固定的 block, 或者在当前批次分配的 block
            if is_pinned or timer == self.timer:
                continue

            self.free_slots.add(slot_id)  # 将 slot 标记为空闲

            # 删除这个映射
            del self.reverse_mapping[slot_id]
            del self.cache[block_key]
            return

        # 如果所有 block 都被固定, 则无法驱逐
        raise RuntimeError("无法驱逐：所有 block 都被固定。请取消一些 block 的固定状态。")

    def _allocate_slot(self) -> int:
        """
        分配一个可用的 slot
        """
        if not self.free_slots:
            raise RuntimeError("No free slots available, should have evicted before allocating")

        # 从空闲集合中取出一个 slot
        slot_id = self.free_slots.pop()
        return slot_id

    def pin_block(self, block_key) -> bool:
        """
        固定一个 block, 防止它被驱逐

        Args:
            block_key: 要固定的 (layer_idx, cpu_block_id)

        Returns:
            bool: 操作是否成功
        """
        if block_key not in self.cache:
            return False

        slot_id, timer, is_pinned = self.cache[block_key]
        if not is_pinned:
            self.cache[block_key] = (slot_id, timer, True)
        return True

    def unpin_block(self, block_key) -> bool:
        """
        取消固定一个 block, 允许它在必要时被驱逐

        Args:
            block_key: 要取消固定的 (layer_idx, cpu_block_id)

        Returns:
            bool: 操作是否成功
        """
        if block_key not in self.cache:
            return False

        slot_id, timer, is_pinned = self.cache[block_key]
        if is_pinned:
            self.cache[block_key] = (slot_id, timer, False)
        return True

class LFUCache(AbstractCache):
    def __init__(self, capacity):
        """
        初始化高效的 LFU Cache, 支持固定 block 功能
        
        Args:
            capacity: Cache 的容量(slot 数量)
        """
        super().__init__(capacity)
        self.cache = {}  # {block_key: (slot_id, frequency, is_pinned, timer)}
        self.reverse_mapping = {}  # {slot_id: block_key}
        self.free_slots = set(range(capacity))
        
        # LFU 核心数据结构：频率分层
        self.freq_to_blocks = {}  # {frequency: {block_key: timer}}}
        self.min_freq = 0  # 当前最小频率
        self.timer = 0

    def add_timer(self):
        self.timer += 1

    def get(self, block_key, hot_score=0.0) -> Tuple[int, bool]:
        """
        获取一个 block, 如果不在 cache 中则加载并分配 slot
        """
        # Cache hit
        if block_key in self.cache:
            slot_id, freq, is_pinned, _ = self.cache[block_key]
            self._update_frequency(block_key, slot_id, freq, is_pinned)
            return slot_id, True

        # Cache miss
        if len(self.cache) >= self.capacity:
            self._evict()

        slot_id = self._allocate_slot()
        
        # 新block频率为1
        self.cache[block_key] = (slot_id, 1, False, self.timer)
        self.reverse_mapping[slot_id] = block_key
        
        # 添加到频率为1的层
        if 1 not in self.freq_to_blocks:
            self.freq_to_blocks[1] = OrderedDict()
        self.freq_to_blocks[1][block_key] = self.timer
        
        # 更新最小频率
        self.min_freq = 1
        
        return slot_id, False

    def _update_frequency(self, block_key, slot_id, old_freq, is_pinned):
        """
        更新block的访问频率
        """
        new_freq = old_freq + 1
        
        # 更新cache中的频率
        self.cache[block_key] = (slot_id, new_freq, is_pinned, self.timer)
        
        # 从旧频率层移除
        del self.freq_to_blocks[old_freq][block_key]
        if not self.freq_to_blocks[old_freq]:
            del self.freq_to_blocks[old_freq]
            # 如果删除的是最小频率层，需要更新min_freq
            if old_freq == self.min_freq:
                self.min_freq += 1

        # 添加到新频率层
        if new_freq not in self.freq_to_blocks:
            self.freq_to_blocks[new_freq] = OrderedDict()
        self.freq_to_blocks[new_freq][block_key] = self.timer

    def _evict(self):
        """
        驱逐使用频率最低的非固定block
        """
        # 从最小频率开始查找可驱逐的block
        while self.min_freq in self.freq_to_blocks:
            freq_dict = self.freq_to_blocks[self.min_freq]
            
            # 在当前频率层中找到第一个非固定的block (FIFO顺序)
            for block_key in list(freq_dict.keys()):
                slot_id, _, is_pinned, timer = self.cache[block_key]
                if not is_pinned and timer != self.timer:
                    # 找到可驱逐的block
                    
                    # 清理所有相关数据结构
                    self.free_slots.add(slot_id)
                    del self.reverse_mapping[slot_id]
                    del self.cache[block_key]
                    del freq_dict[block_key]
                    
                    # 如果当前频率层为空，删除并更新min_freq
                    if not freq_dict:
                        del self.freq_to_blocks[self.min_freq]
                        self._update_min_freq()
                    
                    return
            
            # 当前频率层所有block都被固定，尝试下一个频率
            self.min_freq += 1
            
        # 所有block都被固定
        raise RuntimeError("无法驱逐：所有 block 都被固定。请取消一些 block 的固定状态。")

    def _update_min_freq(self):
        """
        更新最小频率
        """
        if self.freq_to_blocks:
            self.min_freq = min(self.freq_to_blocks.keys())
        else:
            self.min_freq = 0

    def _allocate_slot(self) -> int:
        """
        分配一个可用的slot
        """
        if not self.free_slots:
            raise RuntimeError("No free slots available, should have evicted before allocating")
        return self.free_slots.pop()

    def pin_block(self, block_key) -> bool:
        """
        固定一个block, 防止它被驱逐
        """
        if block_key not in self.cache:
            return False
        
        slot_id, freq, is_pinned, timer = self.cache[block_key]
        if not is_pinned:
            self.cache[block_key] = (slot_id, freq, True, timer)
        return True

    def unpin_block(self, block_key) -> bool:
        """
        取消固定一个block, 允许它在必要时被驱逐
        """
        if block_key not in self.cache:
            return False
            
        slot_id, freq, is_pinned, timer = self.cache[block_key]
        if is_pinned:
            self.cache[block_key] = (slot_id, freq, False, timer)
        return True

class HotScoreCache(AbstractCache):
    def __init__(self, capacity, decay_factor=0.2):
        """
        基于热度分数的高效 Cache, 使用最小堆优化驱逐
        
        Args:
            capacity: Cache 容量
            decay_factor: 热度衰减因子
        """
        super().__init__(capacity)
        self.cache = {}  # {block_key: CacheEntry}
        self.free_slots = set(range(capacity))
        self.decay_factor = decay_factor
        
        # 最小堆：存储 (hot_score, version, block_key, timer)
        self.heap = []
        self.global_version = 0  # 全局版本号，用于懒删除

        self.timer = 0

    def add_timer(self):
        self.timer += 1
        
    def get(self, block_key, hot_score) -> Tuple[int, bool]:
        """获取 block, 基于热度分数管理"""
        
        # Cache hit: 更新热度分数
        if block_key in self.cache:
            entry = self.cache[block_key]
            # 热度衰减 + 新热度
            new_hot_score = entry.hot_score * self.decay_factor + hot_score
            
            # 更新版本号并插入新的堆记录
            self.global_version += 1
            entry.hot_score = new_hot_score
            entry.version = self.global_version
            
            # 将新的热度记录推入堆中
            heapq.heappush(self.heap, (new_hot_score, self.global_version, block_key, self.timer))
            
            return entry.slot_id, True
        
        # Cache miss: 需要加载
        if len(self.cache) >= self.capacity:
            self._evict_by_heap()
            
        slot_id = self._allocate_slot()
        
        # 创建新条目
        self.global_version += 1
        entry = HotCacheEntry(slot_id, hot_score, self.global_version, is_pinned=False, timer=self.timer)
        
        self.cache[block_key] = entry
        
        # 将新block加入堆
        heapq.heappush(self.heap, (hot_score, self.global_version, block_key, self.timer))
        
        return slot_id, False
    
    def _evict_by_heap(self):
        """使用最小堆高效驱逐热度最低的非固定 block"""
        
        while self.heap:
            hot_score, version, block_key, timer = heapq.heappop(self.heap)
            
            # 检查 block 是否还在 cache 中
            if block_key not in self.cache:
                continue  # 懒删除：跳过已被删除的记录
            
            entry = self.cache[block_key]
            
            # 检查版本号是否匹配（懒删除机制）
            if entry.version != version:
                continue  # 跳过过时的记录
            
            # 检查是否被固定
            if entry.is_pinned or timer == self.timer:
                continue  # 跳过固定的 block
            
            # 找到可以驱逐的 block
            self.free_slots.add(entry.slot_id)
            del self.cache[block_key]
            return
        
        # 如果堆空了还没找到可驱逐的block
        raise RuntimeError("无法驱逐：所有 block 都被固定")
    
    def _allocate_slot(self) -> int:
        if not self.free_slots:
            raise RuntimeError("No free slots available")
        return self.free_slots.pop()
    
    def pin_block(self, block_key) -> bool:
        if block_key not in self.cache:
            return False
        self.cache[block_key].is_pinned = True
        return True
    
    def unpin_block(self, block_key) -> bool:
        if block_key not in self.cache:
            return False
        self.cache[block_key].is_pinned = False
        return True
    
    def cleanup_heap(self):
        """
        清理堆中的无效记录，当堆效率过低时调用
        """
        valid_entries = []
        for hot_score, version, block_key, timer in self.heap:
            if (block_key in self.cache and 
                self.cache[block_key].version == version):
                valid_entries.append((hot_score, version, block_key, timer))
        
        self.heap = valid_entries
        heapq.heapify(self.heap)

class HotCacheEntry:
    """Cache 条目，存储热度分数、版本号和元数据"""
    def __init__(self, slot_id, hot_score, version, is_pinned, timer):
        self.slot_id = slot_id
        self.hot_score = hot_score
        self.version = version  # 用于懒删除机制
        self.is_pinned = is_pinned
        self.timer = timer

class LayeredLRUNode:
    """双向链表节点，包含 layer 信息"""
    def __init__(self, slot_id: int, timer: int, is_pinned: bool):
        self.slot_id: int = slot_id
        self.timer: int = timer
        self.is_pinned: bool = is_pinned
        self.prev: Optional['LayeredLRUNode'] = None
        self.next: Optional['LayeredLRUNode'] = None

class LayeredLRUList:
    """每个 layer 的双向链表"""
    def __init__(self):
        # 创建哨兵节点
        self.head = LayeredLRUNode(-1, -1, True)
        self.tail = LayeredLRUNode(-1, -1, True)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0
    
    def add_to_head(self, node: LayeredLRUNode):
        """将节点添加到链表头部（最近使用）"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
        self.size += 1
    
    def remove_node(self, node: LayeredLRUNode):
        """从链表中移除节点"""
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1

    def get_head(self) -> Optional[LayeredLRUNode]:
        """返回链表头部节点（最近使用）"""
        if self.size == 0:
            return None
        return self.head.next
    
    def get_tail(self) -> Optional[LayeredLRUNode]:
        """返回链表尾部节点（最久未使用）"""
        if self.size == 0:
            return None
        return self.tail.prev
    
    def move_to_head(self, node: LayeredLRUNode):
        """将节点移动到链表头部"""
        self.remove_node(node)
        self.add_to_head(node)

class LayerWiseLRUCache(AbstractCache):
    def __init__(self, capacity, num_layers, num_cpu_blocks):
        """
        分层 LRU Cache, 每层独立管理 LRU 列表

        Args:
            capacity: Cache 的容量(slot 数量)
            num_layers: 模型的层数
        """
        super().__init__(capacity)
        self.num_layers = num_layers
        self.num_cpu_blocks = num_cpu_blocks
        # [cpu_block_id -> LayeredLRUNode] per layer
        self.cache_mapping_per_layer = [[LayeredLRUNode(-1, -1, False)
                                            for _ in range(num_cpu_blocks)] 
                                            for _ in range(num_layers)] 
        self.layer_lists: List[LayeredLRUList] = [
            LayeredLRUList() for _ in range(num_layers)
        ]
        self.free_slots = deque(range(capacity))  # 跟踪空闲的 slot
        self.timer = 0

        self.evict_layer_idx = 0  # 用于轮询驱逐

    def add_timer(self):
        self.timer += 1

    def get(self, block_key, hot_score=0.0) -> Tuple[int, bool]:
        """
        获取一个 block, 如果不在 cache 中则加载并分配 slot

        Args:
            block_key: (layer_id, cpu_block_id)
            hot_score: 块热度值

        Returns:
            slot_id: 分配给该 block 的 slot ID
            hit: 是否命中 cache
        """
        layer_idx, cpu_block_id = block_key
        layer_cache_mapping = self.cache_mapping_per_layer[layer_idx]
        layer_list = self.layer_lists[layer_idx]
        
        node = layer_cache_mapping[cpu_block_id]

        # 如果 block 在 cache 中
        if node.slot_id >= 0:
            node.timer = self.timer
            # 更新访问时间和位置(LRU)
            layer_list.move_to_head(node)  # 移到最前面, 表示最近使用
            return node.slot_id, True

        # Cache miss, 需要加载 block

        # 如果 cache 已满, 需要驱逐
        if len(self.free_slots) == 0:
            self._evict(layer_idx)

        # 分配一个新的 slot
        # 将新的 block 加入 cache(默认不固定)
        slot_id = self._allocate_slot() 
        node.slot_id = slot_id
        node.timer = self.timer
        node.is_pinned = False
        layer_list.add_to_head(node)  # 移到最前面

        return slot_id, False

    def _evict(self, layer_idx):
        """
        驱逐最不常用的非固定 block
        """
        _, layer_distance = max((l.size, (idx-layer_idx) % self.num_layers) for idx, l in enumerate(self.layer_lists))

        layer_idx_to_extract = (layer_idx + layer_distance) % self.num_layers

        if layer_idx_to_extract == layer_idx:
            layer_idx_to_extract = (layer_idx_to_extract - 1) % self.num_layers

        num_layers_checked = 0
        while True:
            last_node = self.layer_lists[layer_idx_to_extract].get_tail()
            # 从后向前遍历 (最不常用的在后面)
            # 跳过固定的 block, 或者在当前批次分配的 block
            while last_node is not None and (last_node.timer == self.timer or last_node.is_pinned):
                last_node = last_node.prev
            if last_node is not None:
                break
            layer_idx_to_extract = (layer_idx_to_extract - 1) % self.num_layers
            num_layers_checked += 1
            if num_layers_checked >= self.num_layers:
                raise RuntimeError("无法驱逐：所有 block 都被固定。请取消一些 block 的固定状态。")
            
        self.free_slots.append(last_node.slot_id)  # 将 slot 标记为空闲

        # 删除这个映射
        last_node.slot_id = -1
        last_node.timer = -1
        self.layer_lists[layer_idx_to_extract].remove_node(last_node)
        return


    def _allocate_slot(self) -> int:
        """
        分配一个可用的 slot
        """
        if len(self.free_slots) == 0:
            raise RuntimeError("No free slots available, should have evicted before allocating")

        # 从空闲集合中取出一个 slot
        slot_id = self.free_slots.popleft()
        return slot_id

    def pin_block(self, block_key) -> bool:
        """
        固定一个 block, 防止它被驱逐

        Args:
            block_key: 要固定的 (layer_idx, cpu_block_id)

        Returns:
            bool: 操作是否成功
        """
        layer_idx, cpu_block_id = block_key
        node = self.cache_mapping_per_layer[layer_idx][cpu_block_id]

        if node.slot_id == -1:
            assert "Trying to pin a block that is not in cache"

        node.is_pinned = True
        return True

    def unpin_block(self, block_key) -> bool:
        """
        取消固定一个 block, 允许它在必要时被驱逐

        Args:
            block_key: 要取消固定的 (layer_idx, cpu_block_id)

        Returns:
            bool: 操作是否成功
        """
        layer_idx, cpu_block_id = block_key
        node = self.cache_mapping_per_layer[layer_idx][cpu_block_id]

        if node.slot_id == -1:
            assert "Trying to unpin a block that is not in cache"
        
        node.is_pinned = False
        return True
