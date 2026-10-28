import time
import warnings

import numpy as np
from numba import cuda, int32
from numba.core.errors import NumbaPerformanceWarning

MATRIX_M = 1024
MATRIX_K = 1024
MATRIX_N = 1024
BLOCK_EDGE = 16
LEFT_MODULO = 23
RIGHT_MODULO = 19

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def print_cuda_status() -> None:
    cuda_ready = cuda.is_available()
    print(f"CUDA available: {cuda_ready}")

    if not cuda_ready:
        print("Execution target: CPU fallback")
        return

    print("Execution target: GPU")
    try:
        device = cuda.get_current_device()
        device_name = getattr(device, "name", "Unknown GPU")
        if isinstance(device_name, bytes):
            device_name = device_name.decode()
        print(f"GPU name: {device_name}")
        print(f"Compute capability: {device.compute_capability}")
    except Exception:
        print("GPU detected, but detailed device information is not available in this environment.")


@cuda.jit
def wake_device():
    return


@cuda.jit
def tiled_product_kernel(left_matrix, right_matrix, output_matrix):
    left_tile = cuda.shared.array((16, 16), dtype=int32)
    right_tile = cuda.shared.array((16, 16), dtype=int32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    row_index = cuda.blockIdx.y * BLOCK_EDGE + ty
    col_index = cuda.blockIdx.x * BLOCK_EDGE + tx

    partial_sum = 0
    tile_total = (left_matrix.shape[1] + BLOCK_EDGE - 1) // BLOCK_EDGE

    for tile_id in range(tile_total):
        left_col = tile_id * BLOCK_EDGE + tx
        right_row = tile_id * BLOCK_EDGE + ty

        if row_index < left_matrix.shape[0] and left_col < left_matrix.shape[1]:
            left_tile[ty, tx] = left_matrix[row_index, left_col]
        else:
            left_tile[ty, tx] = 0

        if right_row < right_matrix.shape[0] and col_index < right_matrix.shape[1]:
            right_tile[ty, tx] = right_matrix[right_row, col_index]
        else:
            right_tile[ty, tx] = 0

        cuda.syncthreads()

        for inner_index in range(BLOCK_EDGE):
            partial_sum += left_tile[ty, inner_index] * right_tile[inner_index, tx]

        cuda.syncthreads()

    if row_index < output_matrix.shape[0] and col_index < output_matrix.shape[1]:
        output_matrix[row_index, col_index] = partial_sum


def build_problem(m_dim: int, k_dim: int, n_dim: int) -> tuple[np.ndarray, np.ndarray]:
    left_seed = np.arange(m_dim * k_dim, dtype=np.int32).reshape(m_dim, k_dim)
    right_seed = np.arange(k_dim * n_dim, dtype=np.int32).reshape(k_dim, n_dim)

    left_matrix = ((left_seed * 5) + 11) % LEFT_MODULO
    right_matrix = ((right_seed * 7) + 13) % RIGHT_MODULO
    return left_matrix, right_matrix


def expected_product(left_matrix: np.ndarray, right_matrix: np.ndarray) -> np.ndarray:
    return left_matrix @ right_matrix


def benchmark_gpu_call(kernel_call):
    wall_clock_start = time.perf_counter()
    kernel_seconds = None

    try:
        start_event = cuda.event()
        stop_event = cuda.event()
        start_event.record()
        kernel_call()
        stop_event.record()
        stop_event.synchronize()
        kernel_seconds = cuda.event_elapsed_time(start_event, stop_event) / 1000.0
    except Exception:
        kernel_call()
        cuda.synchronize()

    total_seconds = time.perf_counter() - wall_clock_start
    return total_seconds, kernel_seconds


def run_cpu_version(left_matrix: np.ndarray, right_matrix: np.ndarray, expected: np.ndarray):
    cpu_start = time.perf_counter()
    output_matrix = expected_product(left_matrix, right_matrix)
    total_seconds = time.perf_counter() - cpu_start

    return output_matrix, {
        "mode": "CPU fallback",
        "left_shape": tuple(left_matrix.shape),
        "right_shape": tuple(right_matrix.shape),
        "output_shape": tuple(output_matrix.shape),
        "program_time_s": total_seconds,
        "kernel_time_s": None,
        "validated": bool(np.array_equal(output_matrix, expected)),
    }


def run_gpu_version(left_matrix: np.ndarray, right_matrix: np.ndarray, expected: np.ndarray):
    wake_device[1, 1]()
    cuda.synchronize()

    d_left = cuda.to_device(left_matrix)
    d_right = cuda.to_device(right_matrix)
    d_output = cuda.device_array((left_matrix.shape[0], right_matrix.shape[1]), dtype=left_matrix.dtype)

    threads_per_block = (BLOCK_EDGE, BLOCK_EDGE)
    blocks_per_grid = (
        (right_matrix.shape[1] + BLOCK_EDGE - 1) // BLOCK_EDGE,
        (left_matrix.shape[0] + BLOCK_EDGE - 1) // BLOCK_EDGE,
    )

    tiled_product_kernel[blocks_per_grid, threads_per_block](d_left, d_right, d_output)
    cuda.synchronize()

    total_seconds, kernel_seconds = benchmark_gpu_call(
        lambda: tiled_product_kernel[blocks_per_grid, threads_per_block](d_left, d_right, d_output)
    )

    output_matrix = d_output.copy_to_host()
    return output_matrix, {
        "mode": "CUDA",
        "left_shape": tuple(left_matrix.shape),
        "right_shape": tuple(right_matrix.shape),
        "output_shape": tuple(output_matrix.shape),
        "program_time_s": total_seconds,
        "kernel_time_s": kernel_seconds,
        "validated": bool(np.array_equal(output_matrix, expected)),
    }


def solve_matmul(m_dim: int, k_dim: int, n_dim: int):
    left_matrix, right_matrix = build_problem(m_dim, k_dim, n_dim)
    expected = expected_product(left_matrix, right_matrix)

    if not cuda.is_available():
        return run_cpu_version(left_matrix, right_matrix, expected)

    return run_gpu_version(left_matrix, right_matrix, expected)


def display_report(report: dict) -> None:
    print(f"Execution mode: {report['mode']}")
    print(f"Left shape: {report['left_shape']}")
    print(f"Right shape: {report['right_shape']}")
    print(f"Output shape: {report['output_shape']}")
    print(f"Program time (s): {report['program_time_s']:.6f}")
    if report["kernel_time_s"] is None:
        print("Kernel time (s): n/a")
    else:
        print(f"Kernel time (s): {report['kernel_time_s']:.6f}")
    print(f"Validation: {'OK' if report['validated'] else 'FAILED'}")


def main(m_dim: int = MATRIX_M, k_dim: int = MATRIX_K, n_dim: int = MATRIX_N):
    output_matrix, report = solve_matmul(m_dim, k_dim, n_dim)
    display_report(report)
    if not report["validated"]:
        raise AssertionError("The GPU matrix multiplication result does not match the NumPy reference.")
    return output_matrix, report


if __name__ == "__main__":
    print_cuda_status()
    main()
