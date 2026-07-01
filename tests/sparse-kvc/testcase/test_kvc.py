#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
vLLM KVC Benchmark Pytest Script
Usage: pytest test_kvc.py -v
"""

import json
import logging
import os
import subprocess
import time
from typing import Any, Optional

import pytest
import requests

# ============================================
# 日志配置（pytest 下由 pytest.ini 的 log_cli 负责打屏，勿再 basicConfig）
# ============================================
logger = logging.getLogger(__name__)


# ============================================
# 基础配置 - 从环境变量获取（不变的部分）
# ============================================
class BaseConfig:
    # 路径配置（这些通常不变，仍从环境变量获取）
    MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/models")
    MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen2.5-7B-Instruct-1M")
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

    WORK_DIR = os.environ.get("WORK_DIR", "/vllm-workspace")
    BENCHMARK_DIR = os.environ.get("BENCHMARK_DIR",
                                   os.path.join(WORK_DIR, "vllm"))
    DATASET_DIR = os.environ.get("DATASET_DIR",
                                 os.path.join(WORK_DIR, "LongBench"))
    DATASET_PATH = os.environ.get("DATASET_PATH",
                                  os.path.join(DATASET_DIR, "data.json"))

    # 日志配置
    VLLM_LOG = os.environ.get("VLLM_LOG", os.path.join(WORK_DIR,
                                                       "vllm_run.log"))
    BENCHMARK_LOG = os.environ.get("BENCHMARK_LOG",
                                   os.path.join(WORK_DIR, "run_benchmark.log"))
    PRED_LOG = os.environ.get("PRED_LOG", os.path.join(WORK_DIR,
                                                       "run_pred.log"))

    # 服务配置
    VLLM_PORT = int(os.environ.get("VLLM_PORT", 8001))
    BASE_URL = os.environ.get("BASE_URL", f"http://127.0.0.1:{VLLM_PORT}")

    # 超时设置
    SERVICE_TIMEOUT = int(os.environ.get("SERVICE_TIMEOUT", 600))
    HEALTH_CHECK_INTERVAL = int(os.environ.get("HEALTH_CHECK_INTERVAL", 2))

    # 是否在测试后自动清理
    AUTO_CLEANUP = os.environ.get("AUTO_CLEANUP", "true").lower() == "true"


# ============================================
# 辅助函数
# ============================================
def print_config(
    vllm_params: dict[str, Any],
    benchmark_params: dict[str, Any],
    pred_params: Optional[dict[str, Any]] = None,
):
    """打印当前配置"""
    logger.info("=" * 50)
    logger.info("当前配置:")
    logger.info("  MODEL_PATH: %s", BaseConfig.MODEL_PATH)
    logger.info("  MODEL_NAME: %s", BaseConfig.MODEL_NAME)
    logger.info("  BENCHMARK_DIR: %s", BaseConfig.BENCHMARK_DIR)
    logger.info("  DATASET_DIR: %s", BaseConfig.DATASET_DIR)
    logger.info("  DATASET_PATH: %s", BaseConfig.DATASET_PATH)
    logger.info("  VLLM_PORT: %s", BaseConfig.VLLM_PORT)
    logger.info("  BASE_URL: %s", BaseConfig.BASE_URL)
    logger.info("")
    logger.info("  vLLM参数:")
    for key, value in vllm_params.items():
        logger.info("    %s: %s", key, value)
    logger.info("")
    if benchmark_params:
        logger.info("  Benchmark参数:")
        for key, value in benchmark_params.items():
            logger.info("    %s: %s", key, value)
        logger.info("")
    if pred_params:
        logger.info("  Pred参数:")
        for key, value in pred_params.items():
            logger.info("    %s: %s", key, value)
        logger.info("")
    logger.info("=" * 50)


def check_paths(require_benchmark: bool = True):
    """检查所有路径是否存在"""
    logger.info("检查路径配置...")

    if not os.path.exists(BaseConfig.MODEL_PATH):
        raise FileNotFoundError(f"模型路径不存在: {BaseConfig.MODEL_PATH}")
    logger.info("✓ 模型路径: %s", BaseConfig.MODEL_PATH)

    if require_benchmark:
        benchmark_script = os.path.join(BaseConfig.BENCHMARK_DIR, "benchmarks",
                                        "benchmark_serving.py")
        if not os.path.exists(benchmark_script):
            raise FileNotFoundError(f"Benchmark脚本不存在: {benchmark_script}")
        logger.info("✓ Benchmark路径: %s", BaseConfig.BENCHMARK_DIR)

    if not os.path.exists(BaseConfig.DATASET_PATH):
        raise FileNotFoundError(f"数据集不存在: {BaseConfig.DATASET_PATH}")
    logger.info("✓ 数据集路径: %s", BaseConfig.DATASET_PATH)

    return True


def check_pred_script():
    """检查 pred.py 是否存在"""
    pred_script = os.path.join(BaseConfig.DATASET_DIR, "pred.py")
    if not os.path.exists(pred_script):
        raise FileNotFoundError(f"Pred脚本不存在: {pred_script}")
    logger.info("✓ Pred脚本: %s", pred_script)
    return pred_script


# ============================================
# 核心功能函数
# ============================================
def start_vllm_service(vllm_params: dict[str, Any]) -> subprocess.Popen:
    """
    启动vLLM服务

    Args:
        vllm_params: vLLM参数字典

    Returns:
        subprocess.Popen: 服务进程对象

    Raises:
        RuntimeError: 启动失败时抛出
    """
    logger.info("启动vLLM服务...")

    # 构建vLLM命令
    vllm_cmd = [
        "vllm", "serve", BaseConfig.MODEL_PATH, "--port",
        str(BaseConfig.VLLM_PORT), "--max-model-len",
        str(vllm_params.get("max_model_len",
                            100000)), "--gpu-memory-utilization",
        str(vllm_params.get("gpu_memory_utilization",
                            0.25)), "--enforce-eager", "-tp",
        str(vllm_params.get("tensor_parallel", 4)), "--swap-space",
        str(vllm_params.get("swap_space", 100)), "--enable-chunked-prefill",
        "--max-num-seqs",
        str(vllm_params.get("max_num_seqs", 16)), "--disable-cascade-attn"
    ]
    if vllm_params.get("sparse_topk") is not None:
        vllm_cmd.append("--sparse-topk")
        vllm_cmd.append(str(vllm_params.get("sparse_topk")))
    if vllm_params.get("cache_policy") is not None:
        vllm_cmd.append("--cache-policy")
        vllm_cmd.append(str(vllm_params.get("cache_policy")))
    if vllm_params.get("copy_method") is not None:
        vllm_cmd.append("--copy-method")
        vllm_cmd.append(str(vllm_params.get("copy_method")))

    logger.info("执行命令: %s", " ".join(vllm_cmd))

    # 创建日志目录
    log_dir = os.path.dirname(BaseConfig.VLLM_LOG)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # 启动服务
    with open(BaseConfig.VLLM_LOG, 'w') as log_file:
        process = subprocess.Popen(
            vllm_cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None)

    logger.info("vLLM服务已启动，PID: %s", process.pid)
    logger.info("日志文件: %s", BaseConfig.VLLM_LOG)

    return process


def wait_for_service(timeout: Optional[int] = None) -> bool:
    """
    等待vLLM服务就绪

    Args:
        timeout: 超时时间（秒），默认使用BaseConfig.SERVICE_TIMEOUT

    Returns:
        bool: 服务是否就绪

    Raises:
        TimeoutError: 超时时抛出
    """
    if timeout is None:
        timeout = BaseConfig.SERVICE_TIMEOUT

    max_attempts = timeout // BaseConfig.HEALTH_CHECK_INTERVAL
    attempt = 0

    logger.info("等待vLLM服务就绪 (超时: %s秒)...", timeout)

    while attempt < max_attempts:
        attempt += 1

        # 检查服务是否可访问
        try:
            health_url = BaseConfig.BASE_URL.rstrip("/") + "/health"
            response = requests.get(health_url, timeout=2)
            if response.status_code == 200:
                logger.info("✓ vLLM服务已就绪！(尝试 %s 次)", attempt)
                return True
        except requests.RequestException:
            pass

        # 检查日志中是否有成功标志
        try:
            with open(BaseConfig.VLLM_LOG) as f:
                content = f.read()
                if "Uvicorn running on" in content:
                    logger.info("✓ vLLM服务已就绪！(从日志检测到)")
                    return True
        except (OSError, FileNotFoundError):
            pass

        # 显示进度
        if attempt % 5 == 0:
            logger.info("  等待中... (%s/%s)", attempt, max_attempts)

        time.sleep(BaseConfig.HEALTH_CHECK_INTERVAL)

    # 超时，打印日志
    logger.error("vLLM服务启动超时！")
    try:
        with open(BaseConfig.VLLM_LOG) as f:
            content = f.read()
            logger.error("最后30行日志:")
            lines = content.splitlines()
            for line in lines[-30:]:
                logger.error("  %s", line)
    except (OSError, FileNotFoundError):
        logger.error("无法读取日志文件")

    raise TimeoutError(f"vLLM服务在 {timeout} 秒内未就绪")


def run_benchmark(benchmark_params: dict[str, Any]) -> tuple[int, str]:
    """
    运行Benchmark测试

    Args:
        benchmark_params: Benchmark参数字典

    Returns:
        tuple[int, str]: (退出码, 输出日志)

    Raises:
        RuntimeError: 运行失败时抛出
    """
    logger.info("=" * 50)
    logger.info("开始运行Benchmark测试...")
    logger.info("数据集: %s", BaseConfig.DATASET_PATH)
    logger.info("测试Prompt数: %s", benchmark_params.get("num_prompts", 30))
    logger.info("并发数: %s", benchmark_params.get("max_concurrency", 3))
    min_input_len = benchmark_params.get("min_input_len", 32000)
    max_input_len = benchmark_params.get("max_input_len", 100000)
    logger.info("输入长度范围: %s - %s", min_input_len, max_input_len)
    logger.info("输出长度: %s", benchmark_params.get("output_len", 256))
    logger.info("=" * 50)

    # 切换到benchmark目录
    os.chdir(BaseConfig.BENCHMARK_DIR)

    # 构建benchmark命令
    benchmark_cmd = [
        "python", "benchmarks/benchmark_serving.py", "--backend", "openai",
        "--base-url", BaseConfig.BASE_URL, "--model", BaseConfig.MODEL_PATH,
        "--dataset-name", "custom_jsonl", "--dataset-path",
        BaseConfig.DATASET_PATH, "--num-prompts",
        str(benchmark_params.get("num_prompts", 30)), "--custom-min-input-len",
        str(benchmark_params.get("min_input_len",
                                 32000)), "--custom-max-input-len",
        str(benchmark_params.get("max_input_len",
                                 100000)), "--max-concurrency",
        str(benchmark_params.get("max_concurrency", 3)), "--save-result",
        "--ignore-eos", "--custom-output-len",
        str(benchmark_params.get("output_len", 256))
    ]

    logger.info("执行命令: %s", " ".join(benchmark_cmd))
    logger.info("")

    # 创建日志目录
    log_dir = os.path.dirname(BaseConfig.BENCHMARK_LOG)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # 运行benchmark
    try:
        # 使用subprocess运行并实时输出
        process = subprocess.Popen(benchmark_cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   text=True,
                                   bufsize=1)

        # 收集输出
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line, end='')  # 实时打印
                output_lines.append(line)

        process.wait()
        exit_code = process.returncode

        # 保存日志
        with open(BaseConfig.BENCHMARK_LOG, 'w') as f:
            f.write(''.join(output_lines))

        if exit_code == 0:
            logger.info("")
            logger.info("✓ Benchmark测试完成！")
            result_path = os.path.join(BaseConfig.BENCHMARK_DIR,
                                       "benchmark_results.json")
            logger.info("结果保存在: %s", result_path)
            logger.info("日志保存在: %s", BaseConfig.BENCHMARK_LOG)
        else:
            logger.error("Benchmark测试失败 (退出码: %s)", exit_code)
            logger.error("请检查日志: %s", BaseConfig.BENCHMARK_LOG)

        return exit_code, ''.join(output_lines)

    except Exception as e:
        logger.error("运行Benchmark时出错: %s", e)
        raise RuntimeError(f"Benchmark执行失败: {e}") from e


def run_pred(pred_params: dict[str, Any]) -> tuple[int, str]:
    """
    运行数据集目录下的 pred.py

    Args:
        pred_params: Pred 参数字典，支持 n_proc 等

    Returns:
        tuple[int, str]: (退出码, 输出日志)
    """
    pred_script = check_pred_script()
    n_proc = pred_params.get("n_proc", 3)

    logger.info("=" * 50)
    logger.info("开始运行 Pred 测试...")
    logger.info("脚本: %s", pred_script)
    logger.info("模型: %s", BaseConfig.MODEL_NAME)
    logger.info("并发进程数: %s", n_proc)
    logger.info("=" * 50)

    pred_cmd = [
        "python",
        pred_script,
        "--model",
        BaseConfig.MODEL_NAME,
        "--n_proc",
        str(n_proc),
    ]

    logger.info("执行命令: %s", " ".join(pred_cmd))
    logger.info("")

    log_dir = os.path.dirname(BaseConfig.PRED_LOG)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    try:
        process = subprocess.Popen(
            pred_cmd,
            cwd=BaseConfig.DATASET_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines = []
        for line in iter(process.stdout.readline, ""):
            if line:
                print(line, end="")
                output_lines.append(line)

        process.wait()
        exit_code = process.returncode

        with open(BaseConfig.PRED_LOG, "w") as f:
            f.write("".join(output_lines))

        if exit_code == 0:
            logger.info("")
            logger.info("✓ Pred 测试完成！")
            logger.info("日志保存在: %s", BaseConfig.PRED_LOG)
        else:
            logger.error("Pred 测试失败 (退出码: %s)", exit_code)
            logger.error("请检查日志: %s", BaseConfig.PRED_LOG)

        return exit_code, "".join(output_lines)

    except Exception as e:
        logger.error("运行 Pred 时出错: %s", e)
        raise RuntimeError(f"Pred 执行失败: {e}") from e


def cleanup_vllm_service():
    """清理vLLM服务"""
    if BaseConfig.AUTO_CLEANUP:
        logger.info("停止vLLM服务...")
        try:
            # 使用pkill杀掉vllm进程
            subprocess.run(f"pkill -f 'vllm serve.*{BaseConfig.VLLM_PORT}'",
                           shell=True,
                           capture_output=True)
            time.sleep(2)
            logger.info("vLLM服务已停止")
        except Exception as e:
            logger.warning("停止服务时出错: %s", e)
    else:
        logger.info("vLLM服务继续运行 (AUTO_CLEANUP=false)")


# ============================================
# 各测试用例配置（fixture 在测试体之前运行，需在此集中定义）
# ============================================
TEST_CONFIGS: dict[str, dict[str, dict[str, Any]]] = {
    "test_kvc_hot_perf_1M": {
        "vllm_params": {
            "max_model_len": 1000000,
            "gpu_memory_utilization": 0.25,
            "tensor_parallel": 4,
            "swap_space": 100,
            "copy_method": "raw",
            "sparse_topk": 2048,
            "cache_policy": "hot-score",
            "max_num_seqs": 16,
        },
        "benchmark_params": {
            "num_prompts": 10,
            "min_input_len": 700000,
            "max_input_len": 1000000,
            "max_concurrency": 3,
            "output_len": 256,
        },
    },
    "test_kvc_hot_perf_100k_2048": {
        "vllm_params": {
            "max_model_len": 100000,
            "gpu_memory_utilization": 0.25,
            "tensor_parallel": 4,
            "swap_space": 100,
            "copy_method": "raw",
            "sparse_topk": 2048,
            "cache_policy": "hot-score",
            "max_num_seqs": 16,
        },
        "benchmark_params": {
            "num_prompts": 30,
            "min_input_len": 32000,
            "max_input_len": 100000,
            "max_concurrency": 3,
            "output_len": 256,
        },
    },
    "test_kvc_hot_perf_100k_4096": {
        "vllm_params": {
            "max_model_len": 100000,
            "gpu_memory_utilization": 0.25,
            "tensor_parallel": 4,
            "swap_space": 100,
            "copy_method": "raw",
            "sparse_topk": 4096,
            "cache_policy": "hot-score",
            "max_num_seqs": 16,
        },
        "benchmark_params": {
            "num_prompts": 30,
            "min_input_len": 32000,
            "max_input_len": 100000,
            "max_concurrency": 3,
            "output_len": 256,
        },
    },
    "test_kvc_baseline_perf_1M": {
        "vllm_params": {
            "max_model_len": 1000000,
            "gpu_memory_utilization": 0.7,
            "tensor_parallel": 4,
            "swap_space": 100,
            "max_num_seqs": 16,
        },
        "benchmark_params": {
            "num_prompts": 10,
            "min_input_len": 700000,
            "max_input_len": 1000000,
            "max_concurrency": 3,
            "output_len": 256,
        },
    },
    "test_kvc_baseline_perf_100k": {
        "vllm_params": {
            "max_model_len": 100000,
            "gpu_memory_utilization": 0.25,
            "tensor_parallel": 4,
            "swap_space": 100,
            "max_num_seqs": 16,
        },
        "benchmark_params": {
            "num_prompts": 30,
            "min_input_len": 32000,
            "max_input_len": 100000,
            "max_concurrency": 3,
            "output_len": 256,
        },
    },
    "test_kvc_baseline_acc": {
        "vllm_params": {
            "max_model_len": 1000000,
            "gpu_memory_utilization": 0.7,
            "tensor_parallel": 4,
            "swap_space": 100,
            "max_num_seqs": 16,
        },
        "pred_params": {
            "n_proc": 3,
        },
    },
    "test_kvc_hot_acc_4096": {
        "vllm_params": {
            "max_model_len": 1000000,
            "gpu_memory_utilization": 0.7,
            "copy_method": "raw",
            "sparse_topk": 4096,
            "cache_policy": "hot-score",
            "tensor_parallel": 4,
            "swap_space": 100,
            "max_num_seqs": 16,
        },
        "pred_params": {
            "n_proc": 3,
        },
    },
    "test_kvc_hot_acc_2048": {
        "vllm_params": {
            "max_model_len": 1000000,
            "gpu_memory_utilization": 0.7,
            "copy_method": "raw",
            "sparse_topk": 2048,
            "cache_policy": "hot-score",
            "tensor_parallel": 4,
            "swap_space": 100,
            "max_num_seqs": 16,
        },
        "pred_params": {
            "n_proc": 3,
        },
    },
}


# ============================================
# Pytest Fixture
# ============================================
@pytest.fixture(scope="function")
def vllm_service_with_params(request):
    """
    Pytest fixture: 启动和清理vLLM服务，支持参数传入
    """
    config = TEST_CONFIGS.get(request.node.name, {})
    vllm_params = config.get("vllm_params", {})
    benchmark_params = config.get("benchmark_params", {})
    pred_params = config.get("pred_params", {})
    request.node.benchmark_params = benchmark_params
    request.node.pred_params = pred_params

    # 打印配置
    print_config(vllm_params, benchmark_params, pred_params or None)
    check_paths(require_benchmark=not bool(pred_params))
    if pred_params:
        check_pred_script()

    # 启动服务
    process = start_vllm_service(vllm_params)

    try:
        # 等待服务就绪
        wait_for_service()
        yield process
    finally:
        # Teardown: 清理服务
        cleanup_vllm_service()


# ============================================
# Pytest测试函数
# ============================================
def _assert_benchmark_success(benchmark_params: dict[str, Any]):
    """运行 benchmark 并校验结果"""
    exit_code, _output = run_benchmark(benchmark_params)
    assert exit_code == 0, f"Benchmark执行失败，退出码: {exit_code}"

    result_file = os.path.join(BaseConfig.BENCHMARK_DIR,
                               "benchmark_results.json")
    if os.path.exists(result_file):
        logger.info("✓ 结果文件已生成: %s", result_file)
        try:
            with open(result_file) as f:
                results = json.load(f)
                logger.info("  结果包含 %s 个条目", len(results))
        except Exception as e:
            logger.warning("读取结果文件失败: %s", e)
    else:
        logger.warning("结果文件未生成")


def _assert_pred_success(pred_params: dict[str, Any]):
    """运行 pred.py 并校验结果"""
    exit_code, _output = run_pred(pred_params)
    assert exit_code == 0, f"Pred 执行失败，退出码: {exit_code}"


def _run_named_benchmark_test(request, label: str):
    """按 TEST_CONFIGS 中同名 key 的配置执行 benchmark"""
    benchmark_params = request.node.benchmark_params
    logger.info("=" * 50)
    logger.info("开始执行 %s Benchmark 测试...", label)
    logger.info("=" * 50)
    _assert_benchmark_success(benchmark_params)
    logger.info("=" * 50)
    logger.info("✓ %s 测试通过！", label)
    logger.info("=" * 50)


def _run_named_pred_test(request, label: str):
    """按 TEST_CONFIGS 中同名 key 的配置执行 pred.py"""
    pred_params = request.node.pred_params
    logger.info("=" * 50)
    logger.info("开始执行 %s Pred 测试...", label)
    logger.info("=" * 50)
    _assert_pred_success(pred_params)
    logger.info("=" * 50)
    logger.info("✓ %s Pred 测试通过！", label)
    logger.info("=" * 50)


def test_kvc_hot_perf_1M(vllm_service_with_params, request):
    """KVC hot-score topk=2048，1M 上下文，benchmark 输入 700k~1M."""
    _run_named_benchmark_test(request, "KVC hot-score perf 1M (topk=2048)")


def test_kvc_hot_perf_100k_2048(vllm_service_with_params, request):
    """KVC hot-score topk=2048，100k 上下文，benchmark 输入 32k~100k."""
    _run_named_benchmark_test(request, "KVC hot-score perf 100k (topk=2048)")


def test_kvc_hot_perf_100k_4096(vllm_service_with_params, request):
    """KVC hot-score topk=4096，100k 上下文，benchmark 输入 32k~100k."""
    _run_named_benchmark_test(request, "KVC hot-score perf 100k (topk=4096)")


def test_kvc_baseline_perf_1M(vllm_service_with_params, request):
    """Baseline（无 KVC），1M 上下文，benchmark 输入 700k~1M."""
    _run_named_benchmark_test(request, "Baseline perf 1M")


def test_kvc_baseline_perf_100k(vllm_service_with_params, request):
    """Baseline（无 KVC），100k 上下文，benchmark 输入 32k~100k."""
    _run_named_benchmark_test(request, "Baseline perf 100k")


def test_kvc_baseline_acc(vllm_service_with_params, request):
    """Baseline（无 KVC），1M 上下文，运行 pred.py (n_proc=3)."""
    _run_named_pred_test(request, "Baseline acc 1M")


def test_kvc_hot_acc_4096(vllm_service_with_params, request):
    """KVC hot-score topk=4096，1M 上下文，运行 pred.py (n_proc=3)."""
    _run_named_pred_test(request, "KVC hot-score acc 1M (topk=4096)")


def test_kvc_hot_acc_2048(vllm_service_with_params, request):
    """KVC hot-score topk=2048，1M 上下文，运行 pred.py (n_proc=3)."""
    _run_named_pred_test(request, "KVC hot-score acc 1M (topk=2048)")
