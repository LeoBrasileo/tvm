# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,line-too-long
# ruff: noqa: E501
"""Intrinsics for RISCV Polynomial approximations tensorization"""

from tvm.runtime import DataType
from tvm.script import tirx as T

READ, WRITE = 1, 2 #0b01, 0b10
# float instructions require an explicit rounding mode, use DYN 0b111
mask_args = (T.uint64(0b111),)

def rvv_sigmoid_kernel(
        n_elems: int,
        dtype: str,
        lmul: int,
):
    """Element-wise vector sigmoid using RISC-V vector instructions.

    Args:
        n_elems (int): Number of elements (must fit in lmul vector registers)
        dtype (str): Element dtype, e.g. "float32", "int32"
        lmul (int): LMUL register group multiplier
    """
    # lanes for the scalable vector type
    # (64 // bits) * lmul — LLVM ELEN=64 baseline
    dt = DataType(dtype)
    # scalable min-lanes must stay >= 2: TVM folds `vscale * 1` to a bare
    # vscale and rejects it (bites 64-bit SEW at LMUL=1)
    dtype_lanes = max((64 // dt.bits) * lmul, 2)

    vec_dtype = f"{dtype}xvscalex{dtype_lanes}"
    vec_int_dtype = f"int{dt.bits}xvscalex{dtype_lanes}"

    # ── descriptor ────────────────────────────────────────────────────────────
    @T.prim_func(s_tir=True)
    def rvv_sigmoid_desc(
            A: T.Buffer((n_elems,), dtype, offset_factor=1),
            B: T.Buffer((n_elems,), dtype, offset_factor=1),
    ) -> None:
        with T.sblock("root"):
            T.reads(A[0:n_elems])
            T.writes(B[0:n_elems])
            for i in T.serial(0, n_elems):
                with T.sblock("update"):
                    vi = T.axis.remap("S", [i])
                    B[vi] = T.sigmoid(A[vi])

    # ── implementation ────────────────────────────────────────────────────────

    ln2 = {16: 0.6931472, 32: 0.6931472, 64: 0.6931471805599453}[dt.bits]
    inv_ln2 = {16: 1.4426950, 32: 1.4426950408889634, 64: 1.4426950408889634}[dt.bits]

    vec_one = T.broadcast(getattr(T, dtype)(1.0), T.vscale() * dtype_lanes)
    vec_zero = T.broadcast(getattr(T, dtype)(0.0), T.vscale() * dtype_lanes)
    vec_ln2 = T.broadcast(getattr(T, dtype)(ln2), T.vscale() * dtype_lanes)
    vec_inv_ln2 = T.broadcast(getattr(T, dtype)(inv_ln2), T.vscale() * dtype_lanes)

    # Coefficients for polynomial approximation of exp(-x) using Remez algorithm
    vec_c6 = T.broadcast(getattr(T, dtype)(0.0013888941091214), T.vscale() * dtype_lanes)
    vec_c5 = T.broadcast(getattr(T, dtype)(0.0083333325608882), T.vscale() * dtype_lanes)
    vec_c4 = T.broadcast(getattr(T, dtype)(0.0416666666914562), T.vscale() * dtype_lanes)
    vec_c3 = T.broadcast(getattr(T, dtype)(0.1666666666504284), T.vscale() * dtype_lanes)
    vec_c2 = T.broadcast(getattr(T, dtype)(0.5000000000010834), T.vscale() * dtype_lanes)
    vec_c1 = T.broadcast(getattr(T, dtype)(0.9999999999999124), T.vscale() * dtype_lanes)
    vec_c0 = T.broadcast(getattr(T, dtype)(1.0000000000000002), T.vscale() * dtype_lanes)

    vec_broadcast = T.broadcast(T.Cast(dtype, 0), T.vscale() * dtype_lanes)
    vec_int_zero = T.broadcast(T.Cast(f"int{dt.bits}", 0), T.vscale() * dtype_lanes)

    # fmt: off
    @T.prim_func(s_tir=True)
    def rvv_sigmoid_impl(
            A: T.Buffer((n_elems,), dtype, offset_factor=1),
            B: T.Buffer((n_elems,), dtype, offset_factor=1),
    ) -> None:
        with T.sblock("root"):
            T.reads(A[0:n_elems])
            T.writes(B[0:n_elems])

            vec_A = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vle",
                vec_broadcast,
                T.tvm_access_ptr(T.type_annotation(dtype), A.data, A.elem_offset, n_elems, READ),
                T.int64(n_elems),
            )

            # neg A
            vec_neg_A = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfsub",
                vec_broadcast,
                vec_zero,
                vec_A,
                *mask_args,
                T.int64(n_elems),
            )

            # ── Polynomial approximation of exp() ──────────────────

            # k = -A * inv_ln2
            vec_k = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmul",
                vec_broadcast,
                vec_neg_A,
                vec_inv_ln2,
                *mask_args,
                T.int64(n_elems),
            )

            # k = round(k)
            vec_k_int = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vfcvt.x.f.v",
                vec_int_zero,
                vec_k,
                *mask_args,
                T.int64(n_elems),
            )
            vec_k_float = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfcvt.f.x.v",
                vec_broadcast,
                vec_k_int,
                *mask_args,
                T.int64(n_elems),
            )

            # r = -A - k * ln2
            vec_k_ln2 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmul",
                vec_broadcast,
                vec_k_float,
                vec_ln2,
                *mask_args,
                T.int64(n_elems),
            )
            vec_r = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfsub",
                vec_broadcast,
                vec_neg_A,
                vec_k_ln2,
                *mask_args,
                T.int64(n_elems),
            )

            # Horner
            # p = c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r * c6))))))
            vec_p_6 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                # vec_r * vec_c6 + vec_c5
                vec_r,
                vec_c6,
                vec_c5,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),  # policy (ImmArg, tail-undisturbed)
            )
            vec_p_5 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_r,
                vec_p_6,
                vec_c4,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )
            vec_p_4 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_r,
                vec_p_5,
                vec_c3,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )
            vec_p_3 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_r,
                vec_p_4,
                vec_c2,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )
            vec_p_2 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_r,
                vec_p_3,
                vec_c1,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )
            vec_p = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_r,
                vec_p_2,
                vec_c0,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )

            # 2^k by bit manipulation
            exponent_bias = {16: 15, 32: 127, 64: 1023}[dt.bits]
            mantissa_bits = {16: 10, 32: 23, 64: 52}[dt.bits]
            vec_bias = T.broadcast(T.Cast(f"int{dt.bits}", exponent_bias), T.vscale() * dtype_lanes)
            vec_23 = T.broadcast(T.Cast(f"int{dt.bits}", mantissa_bits), T.vscale() * dtype_lanes)
            vec_k_biased = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vadd",
                vec_int_zero,
                vec_k_int,
                vec_bias,
                T.int64(n_elems),
            )
            vec_k_shifted = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vsll",
                vec_int_zero,
                vec_k_biased,
                vec_23,
                T.int64(n_elems),
            )
            vec_2k = T.reinterpret(vec_dtype, vec_k_shifted)

            # final exp result = p * 2^k
            vec_exp = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmul",
                vec_broadcast,
                vec_p,
                vec_2k,
                *mask_args,
                T.int64(n_elems),
            )

            vec_add = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfadd",
                vec_broadcast,
                vec_exp,
                vec_one,
                *mask_args,
                T.int64(n_elems),
            )
            vec_res = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfdiv",
                vec_broadcast,
                vec_one,
                vec_add,
                *mask_args,
                T.int64(n_elems),
            )

            T.call_llvm_intrin(
                "void",
                "llvm.riscv.vse",
                vec_res,
                T.tvm_access_ptr(T.type_annotation(dtype), B.data, B.elem_offset, n_elems, WRITE),
                T.int64(n_elems),
            )

    # fmt: on
    return rvv_sigmoid_desc, rvv_sigmoid_impl

def rvv_log_kernel(
    n_elems: int,
    dtype: str,
    lmul: int,
):
    """Element-wise vector natural logarithm using RISC-V vector instructions.

    Args:
        n_elems (int): Number of elements (must fit in lmul vector registers)
        dtype (str): Element dtype, e.g. "float32", "float64"
        lmul (int): LMUL register group multiplier
    """
    dt = DataType(dtype)
    dtype_lanes = max((64 // dt.bits) * lmul, 2)

    vec_dtype = f"{dtype}xvscalex{dtype_lanes}"
    vec_int_dtype = f"int{dt.bits}xvscalex{dtype_lanes}"

    # ── descriptor ────────────────────────────────────────────────────────────
    @T.prim_func(s_tir=True)
    def rvv_log_desc(
        A: T.Buffer((n_elems,), dtype, offset_factor=1),
        B: T.Buffer((n_elems,), dtype, offset_factor=1),
    ) -> None:
        with T.sblock("root"):
            T.reads(A[0:n_elems])
            T.writes(B[0:n_elems])
            for i in T.serial(0, n_elems):
                with T.sblock("update"):
                    vi = T.axis.remap("S", [i])
                    B[vi] = T.log(A[vi])

    # ── scalar constants ───────────────────────────────────────────────────────
    ln2 = {16: 0.6931472, 32: 0.69314718056, 64: 0.6931471805599453}[dt.bits]
    one = getattr(T, dtype)(1.0)

    # Minimax polynomial coefficients
    c6 = getattr(T, dtype)(-0.1429207481)
    c5 = getattr(T, dtype)( 0.2205593704)
    c4 = getattr(T, dtype)(-0.2540326919)
    c3 = getattr(T, dtype)( 0.3325802347)
    c2 = getattr(T, dtype)(-0.4999021757)
    c1 = getattr(T, dtype)( 1.0000045588)

    # IEEE-754 layout constants
    exp_bias = {16: 15, 32: 127, 64: 1023}[dt.bits]
    mantissa_shift = {16: 10, 32: 23, 64: 52}[dt.bits]
    exp_mask = {16: 0x1F, 32: 0xFF, 64: 0x7FF}[dt.bits]
    sign_shift = dt.bits - 1

    import struct, math
    if dt.bits == 16:
        # sqrt(2) in float16 bit pattern
        f32_bits = struct.unpack('<H', struct.pack('<e', math.sqrt(2)))[0]
        sqrt2_int = f32_bits
    elif dt.bits == 32:
        sqrt2_int = struct.unpack('<i', struct.pack('<f', math.sqrt(2)))[0]
    else:
        sqrt2_int = struct.unpack('<q', struct.pack('<d', math.sqrt(2)))[0]

    vlanes = T.vscale() * dtype_lanes

    vec_ln2 = T.broadcast(getattr(T, dtype)(ln2), vlanes)
    vec_one = T.broadcast(one, vlanes)

    vec_c6 = T.broadcast(c6, vlanes)
    vec_c5 = T.broadcast(c5, vlanes)
    vec_c4 = T.broadcast(c4, vlanes)
    vec_c3 = T.broadcast(c3, vlanes)
    vec_c2 = T.broadcast(c2, vlanes)
    vec_c1 = T.broadcast(c1, vlanes)

    vec_fp_zero = T.broadcast(T.Cast(dtype, 0), vlanes)
    vec_int_zero = T.broadcast(T.Cast(f"int{dt.bits}", 0), vlanes)
    vec_one_int = T.broadcast(T.Cast(f"int{dt.bits}", 1), vlanes)

    vec_exp_bias_int = T.broadcast(T.Cast(f"int{dt.bits}", exp_bias), vlanes)
    vec_mantissa_shift_int = T.broadcast(T.Cast(f"int{dt.bits}", mantissa_shift), vlanes)
    vec_exp_mask_int = T.broadcast(T.Cast(f"int{dt.bits}", exp_mask), vlanes)
    vec_sign_shift_int = T.broadcast(T.Cast(f"int{dt.bits}", sign_shift), vlanes)
    vec_sqrt2_int = T.broadcast(T.Cast(f"int{dt.bits}", sqrt2_int), vlanes)
    vec_all_ones_int = T.broadcast(T.Cast(f"int{dt.bits}", -1), vlanes)

    # fmt: off
    @T.prim_func(s_tir=True)
    def rvv_log_impl(
        A: T.Buffer((n_elems,), dtype, offset_factor=1),
        B: T.Buffer((n_elems,), dtype, offset_factor=1),
    ) -> None:
        with T.sblock("root"):
            T.reads(A[0:n_elems])
            T.writes(B[0:n_elems])

            vec_A = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vle",
                vec_fp_zero,
                T.tvm_access_ptr(T.type_annotation(dtype), A.data, A.elem_offset, n_elems, READ),
                T.int64(n_elems),
            )

            # Extract biased exponent E
            vec_A_bits = T.reinterpret(vec_int_dtype, vec_A)

            vec_exp_shifted = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vsrl",
                vec_int_zero,
                vec_A_bits,
                vec_mantissa_shift_int,
                T.int64(n_elems),
            )
            vec_Ebiased = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vand",
                vec_int_zero,
                vec_exp_shifted,
                vec_exp_mask_int,
                T.int64(n_elems),
            )
            vec_E = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vsub",
                vec_int_zero,
                vec_Ebiased,
                vec_exp_bias_int,
                T.int64(n_elems),
            )

            # Reconstruct mantissa m in [1, 2)
            # m_bits = (A_bits & ~(exp_mask << mantissa_shift)) | (bias << mantissa_shift)
            vec_exp_mask_shifted = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vsll",
                vec_int_zero,
                vec_exp_mask_int,
                vec_mantissa_shift_int,
                T.int64(n_elems),
            )
            vec_not_exp_mask = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vxor",
                vec_int_zero,
                vec_exp_mask_shifted,
                vec_all_ones_int,
                T.int64(n_elems),
            )
            vec_mantissa_cleared = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vand",
                vec_int_zero,
                vec_A_bits,
                vec_not_exp_mask,
                T.int64(n_elems),
            )
            vec_bias_shifted = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vsll",
                vec_int_zero,
                vec_exp_bias_int,
                vec_mantissa_shift_int,
                T.int64(n_elems),
            )
            vec_m_bits = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vor",
                vec_int_zero,
                vec_mantissa_cleared,
                vec_bias_shifted,
                T.int64(n_elems),
            )

            vec_diff = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vsub",
                vec_int_zero,
                vec_m_bits,
                vec_sqrt2_int,
                T.int64(n_elems),
            )
            # Arithmetic right shift by (bits-1)
            vec_msb = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vsra",
                vec_int_zero,
                vec_diff,
                vec_sign_shift_int,
                T.int64(n_elems),
            )
            # ge_flag = msb + 1
            vec_ge_flag = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vadd",
                vec_int_zero,
                vec_msb,
                vec_one_int,
                T.int64(n_elems),
            )

            # m_adj_bits = m_bits - (ge_flag << mantissa_shift)
            vec_ge_shifted = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vsll",
                vec_int_zero,
                vec_ge_flag,
                vec_mantissa_shift_int,
                T.int64(n_elems),
            )
            vec_m_adj_bits = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vsub",
                vec_int_zero,
                vec_m_bits,
                vec_ge_shifted,
                T.int64(n_elems),
            )
            vec_m_adj = T.reinterpret(vec_dtype, vec_m_adj_bits)

            # E_adj = E + ge_flag
            vec_E_adj = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vadd",
                vec_int_zero,
                vec_E,
                vec_ge_flag,
                T.int64(n_elems),
            )

            # z = m_adj - 1.0 (z in [-0.2929, 0.4142])
            vec_z = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfsub",
                vec_m_adj,
                vec_m_adj,
                vec_one,
                *mask_args,
                T.int64(n_elems),
            )

            # Horner polynomial P(z)
            # tmp0 = z * c6 + c5
            vec_tmp0 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_z,
                vec_c6,
                vec_c5,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )
            # tmp1 = z * tmp0 + c4
            vec_tmp1 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_z,
                vec_tmp0,
                vec_c4,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )
            # tmp2 = z * tmp1 + c3
            vec_tmp2 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_z,
                vec_tmp1,
                vec_c3,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )
            # tmp3 = z * tmp2 + c2
            vec_tmp3 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_z,
                vec_tmp2,
                vec_c2,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )
            # tmp4 = z * tmp3 + c1
            vec_tmp4 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_z,
                vec_tmp3,
                vec_c1,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )
            # P(z) = z * tmp4
            vec_Pz = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmul",
                vec_fp_zero,
                vec_tmp4,
                vec_z,
                *mask_args,
                T.int64(n_elems),
            )

            # Reconstruct: ln(x) = E_adj * ln(2) + P(z)
            vec_E_float = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfcvt.f.x.v",
                vec_fp_zero,
                vec_E_adj,
                *mask_args,
                T.int64(n_elems),
            )

            # result = E_adj_float * ln2 + P(z)
            vec_res = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_E_float,
                vec_ln2,
                vec_Pz,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )

            T.call_llvm_intrin(
                "void",
                "llvm.riscv.vse",
                vec_res,
                T.tvm_access_ptr(T.type_annotation(dtype), B.data, B.elem_offset, n_elems, WRITE),
                T.int64(n_elems),
            )
    # fmt: on
    return rvv_log_desc, rvv_log_impl


def rvv_exp_kernel(
        n_elems: int,
        dtype: str,
        lmul: int,
):
    """Element-wise vector exponential using RISC-V vector instructions.

    Args:
        n_elems (int): Number of elements (must fit in lmul vector registers)
        dtype (str): Element dtype, e.g. "float32", "int32"
        lmul (int): LMUL register group multiplier
    """
    # lanes for the scalable vector type, matching the example's pattern:
    # (64 // bits) * lmul  — LLVM ELEN=64 baseline
    dt = DataType(dtype)
    dtype_lanes = max((64 // dt.bits) * lmul, 2)

    vec_dtype = f"{dtype}xvscalex{dtype_lanes}"
    vec_int_dtype = f"int{dt.bits}xvscalex{dtype_lanes}"

    # ── descriptor ────────────────────────────────────────────────────────────
    @T.prim_func(s_tir=True)
    def rvv_exp_desc(
            A: T.Buffer((n_elems,), dtype, offset_factor=1),
            B: T.Buffer((n_elems,), dtype, offset_factor=1),
    ) -> None:
        with T.sblock("root"):
            T.reads(A[0:n_elems])
            T.writes(B[0:n_elems])
            for i in T.serial(0, n_elems):
                with T.sblock("update"):
                    vi = T.axis.remap("S", [i])
                    B[vi] = T.exp(A[vi])

    # ── implementation ────────────────────────────────────────────────────────

    ln2 = {16: 0.6931472, 32: 0.6931472, 64: 0.6931471805599453}[dt.bits]
    inv_ln2 = {16: 1.4426950, 32: 1.4426950408889634, 64: 1.4426950408889634}[dt.bits]
    vec_ln2 = T.broadcast(getattr(T, dtype)(ln2), T.vscale() * dtype_lanes)
    vec_inv_ln2 = T.broadcast(getattr(T, dtype)(inv_ln2), T.vscale() * dtype_lanes)

    # Coefficients for polynomial approximation of exp(-x) using Remez algorithm
    vec_c6 = T.broadcast(getattr(T, dtype)(0.0013888941091214), T.vscale() * dtype_lanes)
    vec_c5 = T.broadcast(getattr(T, dtype)(0.0083333325608882), T.vscale() * dtype_lanes)
    vec_c4 = T.broadcast(getattr(T, dtype)(0.0416666666914562), T.vscale() * dtype_lanes)
    vec_c3 = T.broadcast(getattr(T, dtype)(0.1666666666504284), T.vscale() * dtype_lanes)
    vec_c2 = T.broadcast(getattr(T, dtype)(0.5000000000010834), T.vscale() * dtype_lanes)
    vec_c1 = T.broadcast(getattr(T, dtype)(0.9999999999999124), T.vscale() * dtype_lanes)
    vec_c0 = T.broadcast(getattr(T, dtype)(1.0000000000000002), T.vscale() * dtype_lanes)

    vec_broadcast = T.broadcast(T.Cast(dtype, 0), T.vscale() * dtype_lanes)
    vec_int_zero = T.broadcast(T.Cast(f"int{dt.bits}", 0), T.vscale() * dtype_lanes)

    # fmt: off
    @T.prim_func(s_tir=True)
    def rvv_exp_impl(
            A: T.Buffer((n_elems,), dtype, offset_factor=1),
            B: T.Buffer((n_elems,), dtype, offset_factor=1),
    ) -> None:
        with T.sblock("root"):
            T.reads(A[0:n_elems])
            T.writes(B[0:n_elems])

            vec_A = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vle",
                vec_broadcast,
                T.tvm_access_ptr(T.type_annotation(dtype), A.data, A.elem_offset, n_elems, READ),
                T.int64(n_elems),
            )

            # ── Polynomial approximation of exp() ──────────────────

            # k = -A * inv_ln2
            vec_k = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmul",
                vec_broadcast,
                vec_A,
                vec_inv_ln2,
                *mask_args,
                T.int64(n_elems),
            )

            # k = round(k)
            vec_k_int = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vfcvt.x.f.v",
                vec_int_zero,
                vec_k,
                *mask_args,
                T.int64(n_elems),
            )
            vec_k_float = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfcvt.f.x.v",
                vec_broadcast,
                vec_k_int,
                *mask_args,
                T.int64(n_elems),
            )

            # r = -A - k * ln2
            vec_k_ln2 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmul",
                vec_broadcast,
                vec_k_float,
                vec_ln2,
                *mask_args,
                T.int64(n_elems),
            )
            vec_r = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfsub",
                vec_broadcast,
                vec_A,
                vec_k_ln2,
                *mask_args,
                T.int64(n_elems),
            )

            # Horner
            # p = c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r * c6))))))
            vec_p_6 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                # vec_r * vec_c6 + vec_c5
                vec_r,
                vec_c6,
                vec_c5,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),  # policy (ImmArg, tail-undisturbed)
            )
            vec_p_5 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_r,
                vec_p_6,
                vec_c4,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )
            vec_p_4 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_r,
                vec_p_5,
                vec_c3,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )
            vec_p_3 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_r,
                vec_p_4,
                vec_c2,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )
            vec_p_2 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_r,
                vec_p_3,
                vec_c1,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )
            vec_p = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_r,
                vec_p_2,
                vec_c0,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )

            # 2^k by bit manipulation
            exponent_bias = {16: 15, 32: 127, 64: 1023}[dt.bits]
            mantissa_bits = {16: 10, 32: 23, 64: 52}[dt.bits]
            vec_bias = T.broadcast(T.Cast(f"int{dt.bits}", exponent_bias), T.vscale() * dtype_lanes)
            vec_23 = T.broadcast(T.Cast(f"int{dt.bits}", mantissa_bits), T.vscale() * dtype_lanes)
            vec_k_biased = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vadd",
                vec_int_zero,
                vec_k_int,
                vec_bias,
                T.int64(n_elems),
            )
            vec_k_shifted = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vsll",
                vec_int_zero,
                vec_k_biased,
                vec_23,
                T.int64(n_elems),
            )
            vec_2k = T.reinterpret(vec_dtype, vec_k_shifted)

            # final exp result = p * 2^k
            vec_exp = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmul",
                vec_broadcast,
                vec_p,
                vec_2k,
                *mask_args,
                T.int64(n_elems),
            )

            T.call_llvm_intrin(
                "void",
                "llvm.riscv.vse",
                vec_exp,
                T.tvm_access_ptr(T.type_annotation(dtype), B.data, B.elem_offset, n_elems, WRITE),
                T.int64(n_elems),
            )

    # fmt: on
    return rvv_exp_desc, rvv_exp_impl


def rvv_tanh_kernel(
        n_elems: int,
        dtype: str,
        lmul: int,
):
    """Element-wise vector hyperbolic tangent using RISC-V vector instructions.

    The implementation rewrites the hyperbolic tangent as
        tanh(x) = 1 - 2 / (exp(2x) + 1)

    Reducing the computation using approximations for exp()

    Args:
        n_elems (int): Number of elements (must fit in lmul vector registers)
        dtype (str): Element dtype, e.g. "float32", "int32"
        lmul (int): LMUL register group multiplier
    """
    # lanes for the scalable vector type, matching the example's pattern:
    # (64 // bits) * lmul  — LLVM ELEN=64 baseline
    dt = DataType(dtype)
    dtype_lanes = max((64 // dt.bits) * lmul, 2)

    vec_dtype = f"{dtype}xvscalex{dtype_lanes}"
    vec_int_dtype = f"int{dt.bits}xvscalex{dtype_lanes}"

    # ── descriptor ────────────────────────────────────────────────────────────
    @T.prim_func(s_tir=True)
    def rvv_tanh_desc(
            A: T.Buffer((n_elems,), dtype, offset_factor=1),
            B: T.Buffer((n_elems,), dtype, offset_factor=1),
    ) -> None:
        with T.sblock("root"):
            T.reads(A[0:n_elems])
            T.writes(B[0:n_elems])
            for i in T.serial(0, n_elems):
                with T.sblock("update"):
                    vi = T.axis.remap("S", [i])
                    B[vi] = T.tanh(A[vi])

    # ── implementation ────────────────────────────────────────────────────────

    ln2 = {16: 0.6931472, 32: 0.6931472, 64: 0.6931471805599453}[dt.bits]
    inv_ln2 = {16: 1.4426950, 32: 1.4426950408889634, 64: 1.4426950408889634}[dt.bits]
    vec_ln2 = T.broadcast(getattr(T, dtype)(ln2), T.vscale() * dtype_lanes)
    vec_inv_ln2 = T.broadcast(getattr(T, dtype)(inv_ln2), T.vscale() * dtype_lanes)
    vec_ones = T.broadcast(getattr(T, dtype)(1), T.vscale() * dtype_lanes)
    vec_twos = T.broadcast(getattr(T, dtype)(2), T.vscale() * dtype_lanes)

    # Coefficients for polynomial approximation of exp(-x) using Remez algorithm
    vec_c6 = T.broadcast(getattr(T, dtype)(0.0013888941091214), T.vscale() * dtype_lanes)
    vec_c5 = T.broadcast(getattr(T, dtype)(0.0083333325608882), T.vscale() * dtype_lanes)
    vec_c4 = T.broadcast(getattr(T, dtype)(0.0416666666914562), T.vscale() * dtype_lanes)
    vec_c3 = T.broadcast(getattr(T, dtype)(0.1666666666504284), T.vscale() * dtype_lanes)
    vec_c2 = T.broadcast(getattr(T, dtype)(0.5000000000010834), T.vscale() * dtype_lanes)
    vec_c1 = T.broadcast(getattr(T, dtype)(0.9999999999999124), T.vscale() * dtype_lanes)
    vec_c0 = T.broadcast(getattr(T, dtype)(1.0000000000000002), T.vscale() * dtype_lanes)

    vec_broadcast = T.broadcast(T.Cast(dtype, 0), T.vscale() * dtype_lanes)
    vec_int_zero = T.broadcast(T.Cast(f"int{dt.bits}", 0), T.vscale() * dtype_lanes)

    # fmt: off
    @T.prim_func(s_tir=True)
    def rvv_tanh_impl(
            A: T.Buffer((n_elems,), dtype, offset_factor=1),
            B: T.Buffer((n_elems,), dtype, offset_factor=1),
    ) -> None:
        with T.sblock("root"):
            T.reads(A[0:n_elems])
            T.writes(B[0:n_elems])

            vec_A = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vle",
                vec_broadcast,
                T.tvm_access_ptr(T.type_annotation(dtype), A.data, A.elem_offset, n_elems, READ),
                T.int64(n_elems),
            )

            vec_2A = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfadd",
                vec_broadcast,
                vec_A,
                vec_A,
                *mask_args,
                T.int64(n_elems),
            )

            # ── Polynomial approximation of exp used for 2x

            # k = -A * inv_ln2
            vec_k = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmul",
                vec_broadcast,
                vec_2A,
                vec_inv_ln2,
                *mask_args,
                T.int64(n_elems),
            )

            # k = round(k)
            vec_k_int = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vfcvt.x.f.v",
                vec_int_zero,
                vec_k,
                *mask_args,
                T.int64(n_elems),
            )
            vec_k_float = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfcvt.f.x.v",
                vec_broadcast,
                vec_k_int,
                *mask_args,
                T.int64(n_elems),
            )

            # r = -A - k * ln2
            vec_k_ln2 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmul",
                vec_broadcast,
                vec_k_float,
                vec_ln2,
                *mask_args,
                T.int64(n_elems),
            )
            vec_r = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfsub",
                vec_broadcast,
                vec_2A,
                vec_k_ln2,
                *mask_args,
                T.int64(n_elems),
            )

            # Horner
            # p = c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r * c6))))))
            vec_p_6 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                # vec_r * vec_c6 + vec_c5
                vec_r,
                vec_c6,
                vec_c5,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),  # policy (ImmArg, tail-undisturbed)
            )
            vec_p_5 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_r,
                vec_p_6,
                vec_c4,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )
            vec_p_4 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_r,
                vec_p_5,
                vec_c3,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )
            vec_p_3 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_r,
                vec_p_4,
                vec_c2,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )
            vec_p_2 = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_r,
                vec_p_3,
                vec_c1,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )
            vec_p = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmadd",
                vec_r,
                vec_p_2,
                vec_c0,
                *mask_args,
                T.int64(n_elems),
                T.int64(0),
            )

            # 2^k by bit manipulation
            exponent_bias = {16: 15, 32: 127, 64: 1023}[dt.bits]
            mantissa_bits = {16: 10, 32: 23, 64: 52}[dt.bits]
            vec_bias = T.broadcast(T.Cast(f"int{dt.bits}", exponent_bias), T.vscale() * dtype_lanes)
            vec_23 = T.broadcast(T.Cast(f"int{dt.bits}", mantissa_bits), T.vscale() * dtype_lanes)
            vec_k_biased = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vadd",
                vec_int_zero,
                vec_k_int,
                vec_bias,
                T.int64(n_elems),
            )
            vec_k_shifted = T.call_llvm_intrin(
                vec_int_dtype,
                "llvm.riscv.vsll",
                vec_int_zero,
                vec_k_biased,
                vec_23,
                T.int64(n_elems),
            )
            vec_2k = T.reinterpret(vec_dtype, vec_k_shifted)

            # final exp result = p * 2^k
            vec_exp = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmul",
                vec_broadcast,
                vec_p,
                vec_2k,
                *mask_args,
                T.int64(n_elems),
            )

            # e^(2x)+1
            vec_denominator = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfadd",
                vec_broadcast,
                vec_exp,
                vec_ones,
                *mask_args,
                T.int64(n_elems)
            )

            # 2/ e^(2x)+1
            vec_div = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfdiv",
                vec_broadcast,
                vec_twos,
                vec_denominator,
                *mask_args,
                T.int64(n_elems)
            )

            # 1 - 2/ e^(2x)+1
            vec_res = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfsub",
                vec_broadcast,
                vec_ones,
                vec_div,
                *mask_args,
                T.int64(n_elems)
            )

            T.call_llvm_intrin(
                "void",
                "llvm.riscv.vse",
                vec_res,
                T.tvm_access_ptr(T.type_annotation(dtype), B.data, B.elem_offset, n_elems, WRITE),
                T.int64(n_elems),
            )

    # fmt: on
    return rvv_tanh_desc, rvv_tanh_impl