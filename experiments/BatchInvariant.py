import contextlib
import torch
import triton
import triton.language as tl
from typing import Any, Dict, Callable

# --- Helper Functions & Metadata ---

def _matmul_launch_metadata(
    grid: Callable[..., Any], kernel: Any, args: Dict[str, Any]
) -> Dict[str, Any]:
    ret = {}
    m, n, k = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}]"
    if "tiles_per_update" in args:
        ret["name"] = (
            f"{kernel.name} [M={m}, N={n}, K={k}, tiles_per_update={args['tiles_per_update']:02}]"
        )
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * m * n * k
    ret["bytes"] = bytes_per_elem * (m * k + n * k + m * n)
    return ret

def get_compute_units():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).multi_processor_count
    if hasattr(torch, "xpu") and torch.xpu.is_available():
         return torch.xpu.get_device_properties(0).max_compute_units
    return torch.get_num_threads()

@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

# --- Triton Kernels ---

@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr,
    A_LARGE: tl.constexpr, B_LARGE: tl.constexpr, C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        
        if A_LARGE: offs_am = offs_am.to(tl.int64)
        if B_LARGE: offs_bn = offs_bn.to(tl.int64)
        
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
            
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
            
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def _log_softmax_kernel(
    input_ptr, output_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    max_val = -float("inf")
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))
        max_val = tl.max(tl.maximum(vals, max_val))

    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    log_sum_exp = tl.log(sum_exp)

    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask)
        output = vals - max_val - log_sum_exp
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)

@triton.jit
def mean_kernel(
    input_ptr, output_ptr, input_stride0, input_stride1, input_stride2,
    output_stride0, output_stride1, M, N, K, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    m_idx = pid // K
    k_idx = pid % K
    if m_idx >= M or k_idx >= K: return

    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N
        input_idx = m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)

    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)

# --- Wrapper Functions ---

def matmul_persistent(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert bias is None or bias.dim() == 1, "Currently assuming bias is 1D"

    NUM_SMS = get_compute_units()
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    def grid(META):
        return (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])),)

    configs = {
        torch.bfloat16: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8},
        torch.float16: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8},
        torch.float32: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8},
    }

    matmul_kernel_persistent[grid](
        a, b, c, bias, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        NUM_SMS=NUM_SMS,
        A_LARGE=a.numel() > 2**31, B_LARGE=b.numel() > 2**31, C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **configs[dtype],
    )
    return c

def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError("This implementation only supports log_softmax along the last dimension")
    
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1]).contiguous()
    n_rows, n_cols = input_2d.shape
    output = torch.empty_like(input_2d)
    BLOCK_SIZE = 1024
    
    _log_softmax_kernel[(n_rows,)](
        input_2d, output, input_2d.stride(0), output.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return output.reshape(original_shape)

def mean_dim(input: torch.Tensor, dim: int, keepdim: bool = False, dtype: torch.dtype | None = None) -> torch.Tensor:
    assert input.is_cuda
    if dim < 0: dim += input.ndim
    if dtype is None:
        dtype = torch.float32 if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64] else input.dtype
    if input.dtype != dtype: input = input.to(dtype)

    shape = list(input.shape)
    M = 1
    for i in range(dim): M *= shape[i]
    N = shape[dim]
    K = 1
    for i in range(dim + 1, len(shape)): K *= shape[i]

    input_3d = input.reshape(M, N, K)
    output_shape = shape.copy() if keepdim else shape[:dim] + shape[dim + 1:]
    if keepdim: output_shape[dim] = 1
    
    output = torch.empty(output_shape, dtype=dtype, device=input.device)
    output_2d = output.reshape(M, K)
    
    mean_kernel[(M * K,)](
        input_3d, output_2d,
        input_3d.stride(0), input_3d.stride(1), input_3d.stride(2),
        output_2d.stride(0), output_2d.stride(1) if output_2d.ndim > 1 else 0,
        M, N, K, BLOCK_SIZE=1024,
    )
    return output

# --- PyTorch Operator Implementations ---

def mm_batch_invariant(a, b):
    return matmul_persistent(a, b)

def addmm_batch_invariant(bias, a, b):
    return matmul_persistent(a, b, bias=bias)

def bmm_batch_invariant(self, mat2):
    # Deterministic Batch Matrix Multiply
    if self.shape[0] == 1:
        return matmul_persistent(self.squeeze(0), mat2.squeeze(0)).unsqueeze(0)
    res = []
    for i in range(self.shape[0]):
        res.append(matmul_persistent(self[i], mat2[i]))
    return torch.stack(res)

def _log_softmax_batch_invariant(input, dim, _half_to_float):
    return log_softmax(input, dim=dim)

def _softmax_batch_invariant(input, dim, dtype=None):
    # Deterministic Softmax
    ls = log_softmax(input, dim=dim)
    return torch.exp(ls)

def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype | None = None):
    assert dtype is None or dtype == torch.float32, f"unsupported dtype: {dtype}"
    if isinstance(dim, int):
        dim = [dim]
    if len(dim) == 1:
        return mean_dim(input, dim[0], keepdim=keepdim)
    else:
        # Fallback for multi-dim reduction
        if len(dim) == 0:
            dim = list(range(input.ndim))
        n_elems = 1
        for d in dim:
            n_elems *= input.shape[d]
        return torch.sum(input, dim=dim, keepdim=keepdim, dtype=torch.float32).to(dtype or input.dtype) / n_elems

def silu_batch_invariant(input):
    # Deterministic SiLU: x * sigmoid(x)
    # We use PyTorch element-wise ops which are generally safe, 
    # but explicit splitting ensures no fused-kernel weirdness.
    return input * torch.sigmoid(input)

# --- Context Manager ---

_batch_invariant_MODE = False
_batch_invariant_LIB = None

def enable_batch_invariant_mode():
    global _batch_invariant_MODE, _batch_invariant_LIB
    if _batch_invariant_MODE: return
    
    if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "current_accelerator"):
         dispatch_key = getattr(torch.accelerator.current_accelerator(), "type", "cpu").upper()
    elif torch.cuda.is_available():
        dispatch_key = "CUDA"
    else:
        dispatch_key = "CPU"
        
    _batch_invariant_MODE = True
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")
    
    # Linear Algebra
    _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, dispatch_key)
    _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, dispatch_key)
    _batch_invariant_LIB.impl("aten::bmm", bmm_batch_invariant, dispatch_key)
    
    # Activation / Normalization
    _batch_invariant_LIB.impl("aten::_log_softmax", _log_softmax_batch_invariant, dispatch_key)
    _batch_invariant_LIB.impl("aten::softmax.int", _softmax_batch_invariant, dispatch_key)
    _batch_invariant_LIB.impl("aten::silu", silu_batch_invariant, dispatch_key)
    
    # Reduction
    _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, dispatch_key)


def disable_batch_invariant_mode():
    global _batch_invariant_MODE, _batch_invariant_LIB
    if _batch_invariant_LIB is not None:
        _batch_invariant_LIB._destroy()
    _batch_invariant_MODE = False
    _batch_invariant_LIB = None

@contextlib.contextmanager
def set_batch_invariant_mode(enabled: bool = True):
    global _batch_invariant_MODE, _batch_invariant_LIB
    old_data = (_batch_invariant_MODE, _batch_invariant_LIB)
    if enabled:
        enable_batch_invariant_mode()
    else:
        disable_batch_invariant_mode()
    yield
    if _batch_invariant_LIB is not None:
        _batch_invariant_LIB._destroy()
    _batch_invariant_MODE, _batch_invariant_LIB = old_data