import time
import warnings

import numpy as np
from numba import cuda, int32
from numba.core.errors import NumbaPerformanceWarning

ROW_COUNT = 6144
COLUMN_COUNT = 8192
TILE_WIDTH = 32
ROW_STRIDE = 8
VALUE_MODULO = 97

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def describe_runtime() -> None:
    gpu_enabled = cuda.is_available()
    print(f"CUDA available: {gpu_enabled}")

    if not gpu_enabled:
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
def warmup_kernel():
    return


@cuda.jit
def shared_transpose_kernel(input_matrix, output_matrix):
    tile = cuda.shared.array((32, 33), dtype=int32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    read_x = cuda.blockIdx.x * TILE_WIDTH + tx
    read_y_base = cuda.blockIdx.y * TILE_WIDTH + ty

    for offset in range(0, 32, 8):
        read_y = read_y_base + offset
        if read_y < input_matrix.shape[0] and read_x < input_matrix.shape[1]:
            tile[ty + offset, tx] = input_matrix[read_y, read_x]

    cuda.syncthreads()

    write_x = cuda.blockIdx.y * TILE_WIDTH + tx
    write_y_base = cuda.blockIdx.x * TILE_WIDTH + ty

    for offset in range(0, 32, 8):
        write_y = write_y_base + offset
        if write_y < output_matrix.shape[0] and write_x < output_matrix.shape[1]:
            output_matrix[write_y, write_x] = tile[tx, ty + offset]


def create_matrix(rows: int, cols: int) -> np.ndarray:
    seed = np.arange(rows * cols, dtype=np.int32).reshape(rows, cols)
    return ((seed * 17) + 3) % VALUE_MODULO


def expected_transpose(matrix: np.ndarray) -> np.ndarray:
    return matrix.T.copy()


def time_cuda_launch(kernel_call):
    timer_start = time.perf_counter()
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

    elapsed_seconds = time.perf_counter() - timer_start
    return elapsed_seconds, kernel_seconds


def execute_on_cpu(source_matrix: np.ndarray, expected: np.ndarray):
    cpu_start = time.perf_counter()
    transposed_matrix = expected_transpose(source_matrix)
    elapsed_seconds = time.perf_counter() - cpu_start

    return transposed_matrix, {
        "mode": "CPU fallback",
        "input_shape": tuple(source_matrix.shape),
        "output_shape": tuple(transposed_matrix.shape),
        "program_time_s": elapsed_seconds,
        "kernel_time_s": None,
        "validated": bool(np.array_equal(transposed_matrix, expected)),
    }


def execute_on_gpu(source_matrix: np.ndarray, expected: np.ndarray):
    warmup_kernel[1, 1]()
    cuda.synchronize()

    d_source = cuda.to_device(source_matrix)
    d_target = cuda.device_array((source_matrix.shape[1], source_matrix.shape[0]), dtype=source_matrix.dtype)

    threads_per_block = (TILE_WIDTH, ROW_STRIDE)
    blocks_per_grid = (
        (source_matrix.shape[1] + TILE_WIDTH - 1) // TILE_WIDTH,
        (source_matrix.shape[0] + TILE_WIDTH - 1) // TILE_WIDTH,
    )

    shared_transpose_kernel[blocks_per_grid, threads_per_block](d_source, d_target)
    cuda.synchronize()

    elapsed_seconds, kernel_seconds = time_cuda_launch(
        lambda: shared_transpose_kernel[blocks_per_grid, threads_per_block](d_source, d_target)
    )

    transposed_matrix = d_target.copy_to_host()
    return transposed_matrix, {
        "mode": "CUDA",
        "input_shape": tuple(source_matrix.shape),
        "output_shape": tuple(transposed_matrix.shape),
        "program_time_s": elapsed_seconds,
        "kernel_time_s": kernel_seconds,
        "validated": bool(np.array_equal(transposed_matrix, expected)),
    }


def solve_transpose(rows: int, cols: int):
    source_matrix = create_matrix(rows, cols)
    expected = expected_transpose(source_matrix)

    if not cuda.is_available():
        return execute_on_cpu(source_matrix, expected)

    return execute_on_gpu(source_matrix, expected)


def print_report(report: dict) -> None:
    print(f"Execution mode: {report['mode']}")
    print(f"Input shape: {report['input_shape']}")
    print(f"Output shape: {report['output_shape']}")
    print(f"Program time (s): {report['program_time_s']:.6f}")
    if report["kernel_time_s"] is None:
        print("Kernel time (s): n/a")
    else:
        print(f"Kernel time (s): {report['kernel_time_s']:.6f}")
    print(f"Validation: {'OK' if report['validated'] else 'FAILED'}")


def main(rows: int = ROW_COUNT, cols: int = COLUMN_COUNT):
    transposed_matrix, report = solve_transpose(rows, cols)
    print_report(report)
    if not report["validated"]:
        raise AssertionError("The GPU transpose result does not match the NumPy reference.")
    return transposed_matrix, report


if __name__ == "__main__":
    describe_runtime()
    main()
