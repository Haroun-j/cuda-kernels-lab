# CUDA kernels lab

Two CUDA kernels in Python with Numba: a tiled matrix multiplication and a shared-memory transpose, checked against NumPy and timed on a Tesla T4. Each comes as a script and as a notebook with recorded outputs.

## What it does

- Multiplies two 1024 x 1024 int32 matrices with 16 x 16 tiles, reading each element once per tile, not sixteen times.
- Transposes a 6144 x 8192 int32 matrix through a padded 32 x 33 tile, with coalesced reads and writes.
- Accepts sizes that are not multiples of the tile: out-of-range threads write zeros.
- Checks both outputs exactly against NumPy; the inputs are small integers, so the int32 sums cannot overflow.

## How it works

```
matmul.py     16 x 16 block, one thread per output -> per K tile: A and B tiles to shared memory
              -> sync -> 16 multiply-adds -> sync -> next tile
transpose.py  32 x 8 block, four rows per thread -> read 32 x 32 tile row-wise -> 32 x 33 shared tile
              -> write transposed rows
```

The padding column shifts each row by one bank, so column-wise reads cause no 32-way bank conflicts, as in NVIDIA's transpose sample.

## Results

One launch on a Colab Tesla T4 (compute capability 7.5), timed with CUDA events after a warm-up.

| kernel | size | kernel time | whole call |
|---|---|---|---|
| tiled matrix multiplication | 1024 x 1024 x 1024 | 6.03 ms | 6.11 ms |
| shared-memory transpose | 6144 x 8192 -> 8192 x 6144 | 2.39 ms | 2.48 ms |

Both outputs match NumPy exactly. The product is 2 x 1024^3 = 2.15 x 10^9 operations, about 356 GOP/s in int32. The transpose moves 201 MB each way, about 168 GB/s. Each time is a single launch, not a median.

## Run it

A CUDA GPU and a Numba CUDA setup: NVIDIA driver and CUDA toolkit, or the `numba-cuda` package.

```bash
pip install -r requirements.txt
python matmul.py            # device name, kernel time, Validation: OK
python transpose.py
```

Without a GPU, `cuda.is_available()` is False and the scripts fall back to NumPy. The timings above come from Colab.

## Limitations

- No naive global-memory version is kept, so the table gives absolute times, not a speedup.
- 16 x 16 tiles with one output per thread leave the T4 well below its peak. Next: larger tiles, more outputs per thread, a copy kernel as bandwidth ceiling, float32 against cuBLAS (CuPy).

Individual lab work for the Calcul GPU module at CY Tech (ING3, AI major, 2025-2026); the course slides are in `docs/`. I wrote the kernels, checks and timing code.
