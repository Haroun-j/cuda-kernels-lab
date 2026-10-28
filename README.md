# CUDA kernels lab: tiled matrix multiplication and transpose

Two CUDA kernels written in Python with Numba for the "Calcul GPU" module at
CY Tech (ING3, 2025-2026). Both use shared-memory tiles, both are checked
against a NumPy reference, and both were timed on the Tesla T4 of a Google
Colab session. Each kernel exists as a notebook (with the recorded outputs)
and as a plain script with the same code.

    matmul.ipynb / matmul.py        1024 x 1024 int32 product, 16 x 16 tiles
    transpose.ipynb / transpose.py  6144 x 8192 int32 transpose, 32 x 32 tiles
    docs/                           course slides (Programmation GPU 2025-2026)

## Matrix multiplication

`tiled_product_kernel` computes C = A x B with one thread per output element,
in blocks of 16 x 16 threads. Reading A and B straight from global memory in
the inner loop would fetch every element of A and B sixteen times per block
and make the accesses to B strided. Instead, each block walks along the K
dimension in steps of 16: it copies a 16 x 16 tile of A and a 16 x 16 tile of
B into shared memory, synchronises, does the sixteen multiply-adds from shared
memory, synchronises again and moves to the next tile. Threads outside the
matrix write zeros into their tile slot, so the kernel also works when the
sizes are not multiples of 16.

The inputs are small integers built with modular arithmetic
(`(5i + 11) mod 23` and `(7i + 13) mod 19`), so the int32 accumulation cannot
overflow and the result can be compared exactly with `A @ B` in NumPy.

## Transpose

`shared_transpose_kernel` moves a 32 x 32 tile through shared memory so that
both the global reads and the global writes are coalesced: a warp reads 32
consecutive elements of an input row and later writes 32 consecutive elements
of an output row, instead of writing a strided column. The tile is declared
32 x 33 rather than 32 x 32: the padding column shifts each row by one bank,
so reading the tile column-wise does not cause 32-way bank conflicts. Blocks
are 32 x 8 threads and each thread handles four rows of the tile, the layout
from NVIDIA's transpose sample, which keeps register use low while still
issuing full 128-byte transactions.

## Measured times

These are the numbers recorded in the notebooks: one kernel launch, measured
with CUDA events after a warm-up launch, on a Tesla T4 (compute capability 7.5).

| kernel                      | size                        | kernel time | whole call |
|-----------------------------|-----------------------------|-------------|------------|
| tiled matrix multiplication | 1024 x 1024 x 1024          | 6.03 ms     | 6.11 ms    |
| shared-memory transpose     | 6144 x 8192 -> 8192 x 6144  | 2.39 ms     | 2.48 ms    |

Both outputs matched the NumPy reference exactly (`Validation: OK`).

For scale: the product is 2 x 1024^3 = 2.15 x 10^9 operations, so 6.03 ms is
about 356 GOP/s in int32. The transpose reads and writes 201 MB each way, so
2.39 ms is about 168 GB/s of effective bandwidth.

I did not keep a naive (global-memory only) version of either kernel in the
notebooks, so there is no measured speedup to quote here, only the absolute
times above.

## Running it

    pip install -r requirements.txt
    python matmul.py
    python transpose.py

With a CUDA GPU and a Numba CUDA setup (NVIDIA driver plus the CUDA toolkit,
or the `numba-cuda` package on recent Numba versions), the scripts print the
device name and the kernel time. Without a GPU, `cuda.is_available()` is
False and the scripts fall back to the NumPy computation; that is how I
checked them on a laptop without a GPU, and the timings above come from Colab
(Runtime > Change runtime type > T4 GPU). The notebooks are the same code
split into cells.

## What I would do next

- Put the naive kernels back next to the tiled ones and measure the actual
  speedup, plus a plain copy kernel as the bandwidth ceiling for the transpose.
- Try 32 x 32 tiles and several outputs per thread for the product; 16 x 16
  tiles leave the T4 well below what it can do.
- Time several launches and report a median instead of a single measurement.
- Switch to float32 and compare against cuBLAS through CuPy.

## Context

"Calcul GPU" module, CY Tech ING3 (AI major), 2025-2026. Individual lab work
by Haroun Joudi, group B. The slides in `docs/` are the course material, not
my work.
