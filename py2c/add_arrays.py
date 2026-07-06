import ctypes
import os
import time

import numpy as np

N = 1_000_000
LIB_DIR = os.path.dirname(os.path.abspath(__file__))

FLOAT_P = ctypes.POINTER(ctypes.c_float)


def to_ptr(arr: np.ndarray):
    return arr.ctypes.data_as(FLOAT_P)


def python_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # plain python loop, no numpy vectorization, to show the "no C/CUDA at all" baseline
    out = [0.0] * len(a)
    for i in range(len(a)):
        out[i] = a[i] + b[i]
    return np.array(out, dtype=np.float32)


def numpy_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def cpp_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lib = ctypes.CDLL(os.path.join(LIB_DIR, "libcpu_add.so"))
    lib.add_arrays_cpp.argtypes = [FLOAT_P, FLOAT_P, FLOAT_P, ctypes.c_int]
    lib.add_arrays_cpp.restype = None

    out = np.empty_like(a)
    lib.add_arrays_cpp(to_ptr(a), to_ptr(b), to_ptr(out), len(a))
    return out


def cuda_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lib = ctypes.CDLL(os.path.join(LIB_DIR, "libcuda_add.so"))
    lib.add_arrays_cuda.argtypes = [FLOAT_P, FLOAT_P, FLOAT_P, ctypes.c_int]
    lib.add_arrays_cuda.restype = None

    out = np.empty_like(a)
    lib.add_arrays_cuda(to_ptr(a), to_ptr(b), to_ptr(out), len(a))
    return out


def timed(fn, *args):
    t0 = time.perf_counter()
    result = fn(*args)
    t1 = time.perf_counter()
    return result, t1 - t0


def main():
    rng = np.random.default_rng(0)
    a = rng.random(N, dtype=np.float64).astype(np.float32)
    b = rng.random(N, dtype=np.float64).astype(np.float32)

    py_out, py_t = timed(python_add, a, b)
    np_out, np_t = timed(numpy_add, a, b)
    cpp_out, cpp_t = timed(cpp_add, a, b)

    # first CUDA call pays for lazy driver/context init (~hundreds of ms),
    # which has nothing to do with the actual add kernel. time it separately
    # from a "warm" call to see the real per-call cost (malloc + H2D/D2H + kernel).
    cuda_out, cuda_cold_t = timed(cuda_add, a, b)
    _, cuda_warm_t = timed(cuda_add, a, b)

    print(f"{'impl':<18}{'time (s)':<12}{'matches numpy'}")
    print(f"{'python':<18}{py_t:<12.4f}{np.allclose(py_out, np_out)}")
    print(f"{'numpy':<18}{np_t:<12.4f}{'-'}")
    print(f"{'cpp':<18}{cpp_t:<12.4f}{np.allclose(cpp_out, np_out)}")
    print(f"{'cuda (cold)':<18}{cuda_cold_t:<12.4f}{np.allclose(cuda_out, np_out)}")
    print(f"{'cuda (warm)':<18}{cuda_warm_t:<12.4f}{'-'}")


if __name__ == "__main__":
    main()
