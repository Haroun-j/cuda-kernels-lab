# CUDA kernels lab: tiled matrix multiplication and transpose

Two CUDA kernels written in Python with Numba for the "Calcul GPU" module at
CY Tech (ING3, AI major, 2025-2026, individual lab work). Both use shared-memory
tiles, are checked against a NumPy reference and were timed on the Tesla T4 of
a Google Colab session. Each kernel exists as a notebook with the recorded
outputs and as a plain script with the same code.

    matmul.ipynb / matmul.py        1024 x 1024 int32 product, 16 x 16 tiles
    transpose.ipynb / transpose.py  6144 x 8192 int32 transpose, 32 x 32 tiles
    docs/                           course slides, Programmation GPU 2025-2026 (course material)

## Matrix multiplication

`tiled_product_kernel` computes C = A x B with one thread per output element,
in blocks of 16 x 16 threads. Each block walks along the K dimension in steps
of 16: it copies a 16 x 16 tile of A and one of B into shared memory,
synchronises, does the sixteen multiply-adds from shared memory, synchronises
again and moves on. Each element of A and B is thus read from global memory
once per tile instead of sixteen times, and the reads of B are contiguous.
Out-of-range threads write zeros into their tile slot, so the sizes need not
be multiples of 16. The inputs are small integers built with modular
arithmetic (`(5i + 11) mod 23` and `(7i + 13) mod 19`), so the int32
accumulation cannot overflow and the result is compared exactly with `A @ B`
in NumPy.

## Transpose

`shared_transpose_kernel` moves a 32 x 32 tile through shared memory so that
both the global reads and writes are coalesced: a warp reads 32 consecutive
elements of an input row and later writes 32 consecutive elements of an
output row instead of a strided column. The tile is declared 32 x 33: the
padding column shifts each row by one bank, so reading it column-wise does
not cause 32-way bank conflicts. Blocks are 32 x 8 threads and each thread
handles four rows of the tile, as in NVIDIA's transpose sample.

## Measured times

One kernel launch, measured with CUDA events after a warm-up launch, on a
Tesla T4 (compute capability 7.5), as recorded in the notebooks.

| kernel                      | size                        | kernel time | whole call |
|-----------------------------|-----------------------------|-------------|------------|
| tiled matrix multiplication | 1024 x 1024 x 1024          | 6.03 ms     | 6.11 ms    |
| shared-memory transpose     | 6144 x 8192 -> 8192 x 6144  | 2.39 ms     | 2.48 ms    |

Both outputs match the NumPy reference exactly (`Validation: OK`). For scale:
the product is 2 x 1024^3 = 2.15 x 10^9 operations, so 6.03 ms is about
356 GOP/s in int32; the transpose reads and writes 201 MB each way, so 2.39 ms
is about 168 GB/s of effective bandwidth.

## Running it

    pip install -r requirements.txt
    python matmul.py
    python transpose.py

The scripts need a CUDA GPU and a Numba CUDA setup (NVIDIA driver and CUDA
toolkit, or the `numba-cuda` package); they print the device name and the
kernel time. Without a GPU, `cuda.is_available()` is False and they fall back
to NumPy. The timings above come from Colab (Runtime > Change runtime type > T4 GPU).

## Limitations

- No naive global-memory version of either kernel is kept, so the table gives
  absolute times, not a speedup.
- Each time is a single launch, not a median over several launches.
- 16 x 16 tiles with one output per thread leave the T4 well below its peak;
  larger tiles, several outputs per thread, a plain copy kernel as bandwidth
  ceiling and a float32 comparison against cuBLAS (CuPy) are the next steps.
