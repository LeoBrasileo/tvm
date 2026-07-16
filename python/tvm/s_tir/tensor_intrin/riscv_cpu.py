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
"""Intrinsics for RISCV tensorization"""

import logging

import tvm_ffi

from tvm.runtime import DataType
from tvm.script import tirx as T
from tvm.target.codegen import Target, llvm_get_vector_width, target_has_features

from .. import TensorIntrin
from .riscv_approximations_cpu import rvv_sigmoid_kernel, rvv_log_kernel, rvv_exp_kernel

logger = logging.getLogger(__name__)

# --- masks and common parameters ---
READ, WRITE = 1, 2 #0b01, 0b10
def mask_llvm(dtype: str):
    """Returns LLVM intrinsic mask arguments for a given dtype."""
    return (T.uint64(0b111),) if dtype.startswith("float") else ()


def get_max_elems(vlen: int, lmul: int, sew: int) -> int:
    """Returns number of elements of a given data type (SEW)
    that fits multiple (LMUL) of the vector registers (VLEN).

    Args:
        vlen (int): VLEN vector length in bits
        lmul (int): LMUL vector lenght multiplier
        sew (int): SEW standard (single) element width

    Returns:
        int: Number of elements
    """
    return (vlen // sew) * lmul


def rvv_vec_dot_product_kernels(
    n_elems: int,
    n_lanes: int,
    data_dtype: str,
    weight_dtype: str,
    out_dtype: str,
    lmul: int,
):
    """Dot product of vector and matrix rows using RISC-V vector instructions.

    These kernels takes two arrays A[ELEMS] and B[ELEMS][MACS] and computes
    dot product of A[ELEMS] with each row of B[LANES], accumulating results
    with C[LANES].

    The pseudo code is as follows:

    .. code-block:: c

        void vec_dot_prod(A[ELEMS], B[LANES][ELEMS], C[LANES]){
            for (j = 0; j < LANES; j++) {
                for (k = 0; k < ELEMS; k++) {
                    C[j] += A[k] * B[j][k]
                }
            }
        }
    """

    @T.prim_func(s_tir=True)
    def rvv_vec_dot_prod_desc(
        A: T.Buffer((n_elems,), data_dtype, offset_factor=1),
        B: T.Buffer((n_elems, n_lanes), weight_dtype, offset_factor=1),
        C: T.Buffer((n_lanes,), out_dtype, offset_factor=1),
    ) -> None:
        with T.sblock("root"):
            T.reads(C[0:n_lanes], A[0:n_elems], B[0:n_elems, 0:n_lanes])
            T.writes(C[0:n_lanes])
            for j in T.serial(0, n_lanes):
                for k in T.serial(0, n_elems):
                    with T.sblock("update"):
                        vj, vk = T.axis.remap("SR", [j, k])
                        C[vj] = C[vj] + T.cast(A[vk], out_dtype) * T.cast(B[vk, vj], out_dtype)

    # LLVM only supports ELEN=32 or ELEN=64
    # https://llvm.org/docs//RISCV/RISCVVectorExtension.html
    d_dtype_lanes = (64 // DataType(data_dtype).bits) * lmul
    w_dtype_lanes = (64 // DataType(weight_dtype).bits) * lmul
    # reduction lanes narrows
    o_dtype_lanes = (64 // DataType(out_dtype).bits) * lmul // n_lanes
    # data type widening case
    o_dtype_lanes = max(o_dtype_lanes, 2)

    wide_dtype = out_dtype
    if DataType(out_dtype).bits > DataType(data_dtype).bits:
        wide_dtype = "".join(c for c in data_dtype if not c.isdigit())
        wide_dtype += str(DataType(data_dtype).bits * 2)

    # fmt: off
    @T.prim_func(s_tir=True)
    def rvv_vec_dot_prod_impl(
        A: T.Buffer((n_elems,), data_dtype, offset_factor=1),
        B: T.Buffer((n_elems, n_lanes), weight_dtype, offset_factor=1, strides=[T.int32(), 1]),
        C: T.Buffer((n_lanes,), out_dtype, offset_factor=1),
    ) -> None:
        with T.sblock("root"):
            T.reads(C[0:n_lanes], A[0:n_elems], B[0:n_elems, 0:n_lanes])
            T.writes(C[0:n_lanes])

            vec_A = T.call_llvm_intrin(
                f"{data_dtype}xvscalex{d_dtype_lanes}",
                "llvm.riscv.vle",
                T.broadcast(T.Cast(data_dtype, 0), T.vscale() * d_dtype_lanes),
                T.tvm_access_ptr(T.type_annotation(data_dtype), A.data, A.elem_offset, n_elems, 1),
                T.int64(n_elems))

            for i in range(n_lanes):
                with T.sblock("update"):
                    T.reads(B[0:n_elems, i])
                    T.writes(C[i])

                    vec_B_row = T.call_llvm_intrin(
                        f"{weight_dtype}xvscalex{w_dtype_lanes}",
                        "llvm.riscv.vlse",
                        T.broadcast(T.Cast(weight_dtype, 0), T.vscale() * w_dtype_lanes),
                        T.tvm_access_ptr(T.type_annotation(weight_dtype), B.data, B.elem_offset + i, n_elems * B.strides[0], 1),
                        T.Cast("int64", B.strides[0] * (DataType(weight_dtype).bits // 8)),
                        T.int64(n_elems))

                    product = T.call_llvm_intrin(
                        f"{wide_dtype}xvscalex{w_dtype_lanes}",
                        "llvm.riscv.vfmul" if out_dtype[0] == "f" else \
                        "llvm.riscv.vwmulsu" if (data_dtype[0] != weight_dtype[0]) else \
                        "llvm.riscv.vwmul",
                        T.broadcast(T.Cast(wide_dtype, 0), T.vscale() * w_dtype_lanes),
                        vec_B_row,
                        vec_A,
                        *mask_llvm(data_dtype),
                        T.uint64(n_elems))

                    ini_acc = T.call_llvm_intrin(
                        f"{out_dtype}xvscalex{o_dtype_lanes}",
                        "llvm.riscv.vle",
                        T.broadcast(T.Cast(out_dtype, 0), T.vscale() * o_dtype_lanes),
                        T.tvm_access_ptr(T.type_annotation(out_dtype), C.data, C.elem_offset + i, 1, 1),
                        T.int64(1))

                    red_sum = T.call_llvm_intrin(
                        f"{out_dtype}xvscalex{o_dtype_lanes}",
                        "llvm.riscv.vfredusum" if out_dtype[0] == "f" else \
                        "llvm.riscv.vwredsum",
                        T.broadcast(T.Cast(out_dtype, 0), T.vscale() * o_dtype_lanes),
                        product,
                        ini_acc,
                        *mask_llvm(data_dtype),
                        T.uint64(n_elems))

                    C[i] = T.call_llvm_intrin(
                        out_dtype,
                        "llvm.riscv.vfmv.f.s" if out_dtype[0] == "f" else \
                        "llvm.riscv.vmv.x.s",
                        red_sum)
    # fmt: on
    return rvv_vec_dot_prod_desc, rvv_vec_dot_prod_impl


def rvv_add_kernel(
    n_elems: int,
    dtype: str,
    lmul: int,
):
    """Element-wise vector add using RISC-V vector instructions.

    Computes C[n_elems] = A[n_elems] + B[n_elems] using vfadd (float)
    or vadd (integer) via explicit LLVM RVV intrinsics, bypassing
    TVM's VectorizeLoop + CreateFAdd path entirely.

    Args:
        n_elems (int): Number of elements (must fit in lmul vector registers)
        dtype   (str): Element dtype, e.g. "float32", "int32"
        lmul    (int): LMUL register group multiplier
    """
    dt = DataType(dtype)
    # TVM cannot encode `vscale * 1` (it folds to a bare vscale),
    # so the min-lane count must stay >= 2.
    dtype_lanes = max((64 // dt.bits) * lmul, 2)

    vec_dtype = f"{dtype}xvscalex{dtype_lanes}"

    # Select intrinsic: vfadd for float, vadd for integer
    is_float = dtype.startswith("float")
    add_intrin = "llvm.riscv.vfadd" if is_float else "llvm.riscv.vadd"

    # ── descriptor ────────────────────────────────────────────────────────────
    @T.prim_func(s_tir=True)
    def rvv_add_desc(
        A: T.Buffer((n_elems,), dtype, offset_factor=1),
        B: T.Buffer((n_elems,), dtype, offset_factor=1),
        C: T.Buffer((n_elems,), dtype, offset_factor=1),
    ) -> None:
        with T.sblock("root"):
            T.reads(A[0:n_elems], B[0:n_elems])
            T.writes(C[0:n_elems])
            for i in T.serial(0, n_elems):
                with T.sblock("update"):
                    vi = T.axis.remap("S", [i])
                    C[vi] = A[vi] + B[vi]

    # ── implementation ────────────────────────────────────────────────────────
    # fmt: off
    @T.prim_func(s_tir=True)
    def rvv_add_impl(
        A: T.Buffer((n_elems,), dtype, offset_factor=1),
        B: T.Buffer((n_elems,), dtype, offset_factor=1),
        C: T.Buffer((n_elems,), dtype, offset_factor=1),
    ) -> None:
        with T.sblock("root"):
            T.reads(A[0:n_elems], B[0:n_elems])
            T.writes(C[0:n_elems])

            vec_A = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vle",
                T.broadcast(T.Cast(dtype, 0), T.vscale() * dtype_lanes),
                T.tvm_access_ptr(T.type_annotation(dtype), A.data, A.elem_offset, n_elems, READ),
                T.int64(n_elems),
            )
            vec_B = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vle",
                T.broadcast(T.Cast(dtype, 0), T.vscale() * dtype_lanes),
                T.tvm_access_ptr(T.type_annotation(dtype), B.data, B.elem_offset, n_elems, READ),
                T.int64(n_elems),
            )

            vec_C = T.call_llvm_intrin(
                vec_dtype,
                add_intrin,
                T.broadcast(T.Cast(dtype, 0), T.vscale() * dtype_lanes),
                vec_A,
                vec_B,
                *mask_llvm(dtype),
                T.int64(n_elems),
            )

            T.call_llvm_intrin(
                "void",
                "llvm.riscv.vse",
                vec_C,
                T.tvm_access_ptr(T.type_annotation(dtype), C.data, C.elem_offset, n_elems, WRITE),
                T.int64(n_elems),
            )
    # fmt: on
    return rvv_add_desc, rvv_add_impl


def rvv_copy_kernel(n_elems: int, dtype: str, lmul: int):
    """Implementation of a plain scalar copy loop over n_elems elements
    for concatenating using RISC-V vector instructions.

    The descriptor is a plain scalar copy loop over n_elems elements.
    The implementation replaces it with a single vle + vse pair.
    Two-buffer signature (A -> C).
    """
    dt = DataType(dtype)
    dtype_lanes = max((64 // dt.bits) * lmul, 2)
    vec_dtype = f"{dtype}xvscalex{dtype_lanes}"

    @T.prim_func(s_tir=True)
    def rvv_copy_desc(
            A: T.Buffer((n_elems,), dtype, offset_factor=1),
            C: T.Buffer((n_elems,), dtype, offset_factor=1),
    ) -> None:
        with T.sblock("root"):
            T.reads(A[0:n_elems])
            T.writes(C[0:n_elems])
            for i in T.serial(0, n_elems):
                with T.sblock("update"):
                    vi = T.axis.remap("S", [i])
                    C[vi] = A[vi]

    # fmt: off
    @T.prim_func(s_tir=True)
    def rvv_copy_impl(
            A: T.Buffer((n_elems,), dtype, offset_factor=1),
            C: T.Buffer((n_elems,), dtype, offset_factor=1),
    ) -> None:
        with T.sblock("root"):
            T.reads(A[0:n_elems])
            T.writes(C[0:n_elems])

            vec_A = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vle",
                T.broadcast(T.Cast(dtype, 0), T.vscale() * dtype_lanes),
                T.tvm_access_ptr(T.type_annotation(dtype), A.data, A.elem_offset, n_elems, READ),
                T.int64(n_elems),
            )
            T.call_llvm_intrin(
                "void",
                "llvm.riscv.vse",
                vec_A,
                T.tvm_access_ptr(T.type_annotation(dtype), C.data, C.elem_offset, n_elems, WRITE),
                T.int64(n_elems),
            )

    # fmt: on

    return rvv_copy_desc, rvv_copy_impl


def rvv_add_relu_kernel(
        n_elems: int,
        dtype: str,
        lmul: int,
):
    """Element-wise vector add using RISC-V vector instructions.

    Computes C[n_elems] = Max(A[n_elems] + B[n_elems], 0) using explicit LLVM RVV intrinsics

    Args:
        n_elems (int): Number of elements (must fit in lmul vector registers)
        dtype   (str): Element dtype, e.g. "float32", "int32"
        lmul    (int): LMUL register group multiplier
    """
    dt = DataType(dtype)
    # TVM cannot encode `vscale * 1` (it folds to a bare vscale),
    # so the min-lane count must stay >= 2.
    dtype_lanes = max((64 // dt.bits) * lmul, 2)

    vec_dtype = f"{dtype}xvscalex{dtype_lanes}"

    # Select intrinsic: vfadd for float, vadd for integer
    is_float = dtype.startswith("float")
    add_intrin = "llvm.riscv.vfadd" if is_float else "llvm.riscv.vadd"

    # ── descriptor ────────────────────────────────────────────────────────────
    @T.prim_func(s_tir=True)
    def rvv_add_relu_desc(
            A: T.Buffer((n_elems,), dtype, offset_factor=1),
            B: T.Buffer((n_elems,), dtype, offset_factor=1),
            C: T.Buffer((n_elems,), dtype, offset_factor=1),
    ) -> None:
        with T.sblock("root"):
            T.reads(A[0:n_elems], B[0:n_elems])
            T.writes(C[0:n_elems])
            for i in T.serial(0, n_elems):
                with T.sblock("update"):
                    vi = T.axis.remap("S", [i])
                    C[vi] = T.max(A[vi] + B[vi], 0)

    # ── implementation ────────────────────────────────────────────────────────
    # fmt: off
    @T.prim_func(s_tir=True)
    def rvv_add_relu_impl(
            A: T.Buffer((n_elems,), dtype, offset_factor=1),
            B: T.Buffer((n_elems,), dtype, offset_factor=1),
            C: T.Buffer((n_elems,), dtype, offset_factor=1),
    ) -> None:
        with T.sblock("root"):
            T.reads(A[0:n_elems], B[0:n_elems])
            T.writes(C[0:n_elems])

            vec_A = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vle",
                T.broadcast(T.Cast(dtype, 0), T.vscale() * dtype_lanes),
                T.tvm_access_ptr(T.type_annotation(dtype), A.data, A.elem_offset, n_elems, READ),
                T.int64(n_elems),
            )
            vec_B = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vle",
                T.broadcast(T.Cast(dtype, 0), T.vscale() * dtype_lanes),
                T.tvm_access_ptr(T.type_annotation(dtype), B.data, B.elem_offset, n_elems, READ),
                T.int64(n_elems),
            )

            vec_C = T.call_llvm_intrin(
                vec_dtype,
                add_intrin,
                T.broadcast(T.Cast(dtype, 0), T.vscale() * dtype_lanes),
                vec_A,
                vec_B,
                *mask_llvm(dtype),
                T.int64(n_elems),
            )

            vec_relu = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfmax" if is_float else "llvm.riscv.vmax",
                T.broadcast(T.Cast(dtype, 0), T.vscale() * dtype_lanes),
                vec_C,
                T.broadcast(T.Cast(dtype, 0), T.vscale() * dtype_lanes),
                T.int64(n_elems),
            )

            T.call_llvm_intrin(
                "void",
                "llvm.riscv.vse",
                vec_relu,
                T.tvm_access_ptr(T.type_annotation(dtype), C.data, C.elem_offset, n_elems, WRITE),
                T.int64(n_elems),
            )
    # fmt: on
    return rvv_add_relu_desc, rvv_add_relu_impl


def rvv_copy_kernel(n_elems: int, dtype: str, lmul: int):
    """Implementation of a plain scalar copy loop over n_elems elements
    for concatenating using RISC-V vector instructions.

    The descriptor is a plain scalar copy loop over n_elems elements.
    The implementation replaces it with a single vle + vse pair.
    Two-buffer signature (A -> C).
    """
    dt = DataType(dtype)
    dtype_lanes = max((64 // dt.bits) * lmul, 2)
    vec_dtype = f"{dtype}xvscalex{dtype_lanes}"

    @T.prim_func(s_tir=True)
    def rvv_copy_desc(
            A: T.Buffer((n_elems,), dtype, offset_factor=1),
            C: T.Buffer((n_elems,), dtype, offset_factor=1),
    ) -> None:
        with T.sblock("root"):
            T.reads(A[0:n_elems])
            T.writes(C[0:n_elems])
            for i in T.serial(0, n_elems):
                with T.sblock("update"):
                    vi = T.axis.remap("S", [i])
                    C[vi] = A[vi]

    # fmt: off
    @T.prim_func(s_tir=True)
    def rvv_copy_impl(
            A: T.Buffer((n_elems,), dtype, offset_factor=1),
            C: T.Buffer((n_elems,), dtype, offset_factor=1),
    ) -> None:
        with T.sblock("root"):
            T.reads(A[0:n_elems])
            T.writes(C[0:n_elems])

            vec_A = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vle",
                T.broadcast(T.Cast(dtype, 0), T.vscale() * dtype_lanes),
                T.tvm_access_ptr(T.type_annotation(dtype), A.data, A.elem_offset, n_elems, READ),
                T.int64(n_elems),
            )
            T.call_llvm_intrin(
                "void",
                "llvm.riscv.vse",
                vec_A,
                T.tvm_access_ptr(T.type_annotation(dtype), C.data, C.elem_offset, n_elems, WRITE),
                T.int64(n_elems),
            )

    # fmt: on

    return rvv_copy_desc, rvv_copy_impl

def rvv_neg_kernel(
    n_elems: int,
    dtype: str,
    lmul: int,
):
    """Element-wise vector negative using RISC-V vector instructions.

    Args:
        n_elems (int): Number of elements (must fit in lmul vector registers)
        dtype (str): Element dtype, e.g. "float32", "int32"
        lmul (int): LMUL register group multiplier
    """
    dt = DataType(dtype)
    dtype_lanes = max((64 // dt.bits) * lmul, 2)

    # for uint input, the signed counterpart is used for output and vsub
    is_uint = dtype.startswith("uint")
    signed_dtype = f"int{dt.bits}" if is_uint else dtype
    out_dtype = signed_dtype

    vec_dtype = f"{dtype}xvscalex{dtype_lanes}"

    # ── descriptor ────────────────────────────────────────────────────────────
    @T.prim_func(s_tir=True)
    def rvv_neg_desc(
        A: T.Buffer((n_elems,), dtype, offset_factor=1),
        B: T.Buffer((n_elems,), out_dtype, offset_factor=1),
    ) -> None:
        with T.sblock("root"):
            T.reads(A[0:n_elems])
            T.writes(B[0:n_elems])
            for i in T.serial(0, n_elems):
                with T.sblock("update"):
                    vi = T.axis.remap("S", [i])
                    if is_uint:
                        B[vi] = T.Cast(out_dtype, A[vi]) * getattr(T, out_dtype)(-1)
                    else:
                        B[vi] = A[vi] * getattr(T, dtype)(-1)

    # ── implementation ────────────────────────────────────────────────────────
    vec_broadcast = T.broadcast(T.Cast(signed_dtype, 0), T.vscale() * dtype_lanes)

    # fmt: off
    @T.prim_func(s_tir=True)
    def rvv_neg_impl(
        A: T.Buffer((n_elems,), dtype, offset_factor=1),
        B: T.Buffer((n_elems,), out_dtype, offset_factor=1),
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

            vec_neg_A = T.call_llvm_intrin(
                vec_dtype,
                "llvm.riscv.vfsub" if dtype.startswith("float") else "llvm.riscv.vsub",
                T.broadcast(getattr(T, signed_dtype)(0), T.vscale() * dtype_lanes),
                T.broadcast(getattr(T, signed_dtype)(0), T.vscale() * dtype_lanes),
                vec_A,
                *mask_llvm(signed_dtype),
                T.int64(n_elems),
            )

            T.call_llvm_intrin(
                "void",
                "llvm.riscv.vse",
                vec_neg_A,
                T.tvm_access_ptr(T.type_annotation(out_dtype), B.data, B.elem_offset, n_elems, WRITE),
                T.int64(n_elems),
            )
    # fmt: on
    return rvv_neg_desc, rvv_neg_impl


@tvm_ffi.register_global_func("tirx.tensor_intrin.register_rvv_isa_intrinsics")
def register_rvv_isa_intrinsics(target: Target, inventory_only=False) -> dict():
    """Register RISCV V (vector) intrinsics
    [x] Implementation follows version 1.0 vector specifications:
        https://github.com/riscvarchive/riscv-v-spec/releases/tag/v1.0

    Args:
        target (Target): TVM target
        inventory_only (bool): No registration inventory only

    Returns:
        dict(): A catalog with registered kernel names and properties
    """
    if not target_has_features("v", target):
        raise RuntimeError("Current target does not support `v` extension.")

    vlen = llvm_get_vector_width(target)
    # get maximum reduction lanes (without grouping)
    n_lanes = get_max_elems(vlen, lmul=1, sew=32)

    kernels_inventory = {}

    data_dtype = ["uint8", "int8", "float16", "float32"]
    weight_dtype = ["int8", "int8", "float16", "float32"]
    output_dtype = ["int32", "int32", "float16", "float32"]

    for d_dtype, w_dtype, o_dtype in zip(data_dtype, weight_dtype, output_dtype):
        # max elements to grouped registers
        max_elems = get_max_elems(vlen, lmul=8, sew=DataType(d_dtype).bits)
        # data widening halves available vector registers
        if DataType(o_dtype).bits > DataType(d_dtype).bits:
            max_elems //= 2
        # compute optimal LMUL for full load
        lmul = max_elems // (vlen // DataType(d_dtype).bits)

        n_elems = max_elems
        while n_elems >= 4:
            dt = DataType(d_dtype)
            wt = DataType(w_dtype)
            ot = DataType(o_dtype)
            kernel_name = "rvv_dot"
            kernel_name += f"_{n_elems}{dt[0]}{dt.bits}"
            kernel_name += f"_{n_lanes}x{n_elems}{wt[0]}{wt.bits}"
            kernel_name += f"_{n_lanes}{ot[0]}{ot.bits}"
            kernels_inventory[kernel_name] = n_elems

            if not inventory_only:
                logger.debug(f"Registering kernel {kernel_name}")
                desc, impl = rvv_vec_dot_product_kernels(
                    n_elems, n_lanes, d_dtype, w_dtype, o_dtype, lmul
                )
                TensorIntrin.register(kernel_name, desc, impl, override=True)

            n_elems //= 2

    return kernels_inventory

@tvm_ffi.register_global_func("tirx.tensor_intrin.register_rvv_isa_spatial_intrinsics")
def register_rvv_isa_spatial_intrinsics(target: Target, inventory_only=False) -> dict():
    """Register RISCV V (vector) Spatial intrinsics.
    These include classic one buffer kernels and polynomial approximations

    Args:
        target (Target): TVM target
        inventory_only (bool): No registration inventory only

    Returns:
        dict(): A catalog with registered kernel names and properties
    """
    if not target_has_features("v", target):
        raise RuntimeError("Current target does not support `v` extension.")

    vlen = llvm_get_vector_width(target)
    kernels_inventory = {}

    # ── Spatial Kernels ────────────────────────────────────────────────────────
    dtypes = [#"uint8", "uint16", "uint32", "uint64",
              #"int8", "int32", "int64", #"int16", "float16",
              "float32", "float64"]
    for dtype in dtypes:
        dt = DataType(dtype)
        max_elems = get_max_elems(vlen, lmul=8, sew=dt.bits)

        n_elems = max_elems
        while n_elems >= 4:
            # size LMUL to the tile so the register group matches n_elems
            lmul = max(1, n_elems // (vlen // dt.bits))
            kernel_add_name = f"rvv_add_{n_elems}{dt[0]}{dt.bits}"
            kernel_add_relu_name = f"rvv_add_relu_{n_elems}{dt[0]}{dt.bits}"
            #kernel_concat_name = f"rvv_copy_{n_elems}{dt[0]}{dt.bits}"
            kernel_neg_name = f"rvv_neg_{n_elems}{dt[0]}{dt.bits}"
            for k in (kernel_add_relu_name, kernel_add_name, kernel_neg_name):
                kernels_inventory[k] = n_elems

            if not inventory_only:
                for kernel_name, kernel_fn in [
                    (kernel_add_relu_name, rvv_add_relu_kernel),
                    (kernel_add_name, rvv_add_kernel),
                    #(kernel_concat_name, rvv_copy_kernel),
                    (kernel_neg_name, rvv_neg_kernel),
                ]:
                    logger.debug(f"Registering kernel {kernel_name}")
                    desc, impl = kernel_fn(n_elems, dtype, lmul)
                    TensorIntrin.register(kernel_name, desc, impl, override=True)

            n_elems //= 2

    # ── Polynomial Approximation Kernels ────────────────────────────────────────────────────────
    # These kernels work only on float values and are polynomial approximations implemented fully RVV
    #TODO: float16 causes a SEGFAULT on LLVM, needs further research
    dtypes = ["float32", "float64"]
    for dtype in dtypes:
        dt = DataType(dtype)
        max_elems = get_max_elems(vlen, lmul=8, sew=dt.bits)

        n_elems = max_elems
        while n_elems >= 4:
            # size LMUL to the tile so the register group matches n_elems
            lmul = max(1, n_elems // (vlen // dt.bits))
            kernel_sigmoid_name = f"rvv_sigmoid_{n_elems}{dt[0]}{dt.bits}"
            kernel_log_name = f"rvv_log_{n_elems}{dt[0]}{dt.bits}"
            kernel_exp_name = f"rvv_exp_{n_elems}{dt[0]}{dt.bits}"
            for k in (kernel_sigmoid_name, kernel_log_name, kernel_exp_name):
                kernels_inventory[k] = n_elems

            if not inventory_only:
                for kernel_name, kernel_fn in [
                    (kernel_sigmoid_name, rvv_sigmoid_kernel),
                    (kernel_log_name, rvv_log_kernel),
                    (kernel_exp_name, rvv_exp_kernel),
                ]:
                    logger.debug(f"Registering kernel {kernel_name}")
                    desc, impl = kernel_fn(n_elems, dtype, lmul)
                    TensorIntrin.register(kernel_name, desc, impl, override=True)

            n_elems //= 2

    return kernels_inventory


def register_riscv_intrinsics(target: Target):
    """Register RISCV intrinsics

    Args:
        target (Target): TVM target
    """

    # RISCV `v` 1.0 extension templates
    _ = register_rvv_isa_intrinsics(target)
    _ = register_rvv_isa_spatial_intrinsics(target)
    logger.debug("Finished registering riscv intrinsics.")