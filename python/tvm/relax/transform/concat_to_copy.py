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
# pylint: disable=invalid-name, unused-argument, redefined-argument-from-local
"""Relax ConcatToCopy pass."""

import tvm
import tvm.relax as relax
from tvm import tirx
from tvm import IRModule
from tvm.relax.expr_functor import PyExprMutator, mutator


def normalize_mod(mod):
    """Force the same normalization the parser roundtrip applies."""
    src = mod.script()
    return tvm.script.from_source(src)


def make_concat_copy_func(shapes, out_shape, axis, dtype="float32"):
    ndim = len(out_shape)
    inp_bufs = []
    for i, s in enumerate(shapes):
        buf = tirx.decl_buffer(s, dtype, name=f"inp{i}")
        inp_bufs.append(buf)
    out_buf = tirx.decl_buffer(out_shape, dtype, name="out")

    stmts = []
    offset = 0
    for i, (inp_buf, s) in enumerate(zip(inp_bufs, shapes)):
        loop_vars = [tirx.Var(f"ax{d}", "int32") for d in range(ndim)]
        block_vars = [tirx.Var(f"v_ax{d}", "int32") for d in range(ndim)]

        out_indices = list(block_vars)
        out_indices[axis] = block_vars[axis] + tirx.const(offset, "int32")

        store = tirx.BufferStore(out_buf, tirx.BufferLoad(inp_buf, block_vars), out_indices)

        update_iter_vars = [
            tirx.IterVar(
                tvm.ir.Range.from_min_extent(tirx.const(0, "int32"), tirx.const(s[d], "int32")),
                block_vars[d],
                tirx.IterVar.DataPar,
            )
            for d in range(ndim)
        ]
        update_reads = [
            tirx.BufferRegion(
                inp_buf,
                [tvm.ir.Range.from_min_extent(block_vars[d], tirx.const(1, "int32")) for d in range(ndim)],
            )
        ]
        update_writes = [
            tirx.BufferRegion(
                out_buf,
                [tvm.ir.Range.from_min_extent(out_indices[d], tirx.const(1, "int32")) for d in range(ndim)],
            )
        ]

        inner_block = tirx.SBlock(
            iter_vars=update_iter_vars,
            reads=update_reads,
            writes=update_writes,
            name_hint=f"T_concat_copy_{i}",
            body=store,
            alloc_buffers=[],
        )
        inner_realize = tirx.SBlockRealize(
            iter_values=loop_vars,
            predicate=tirx.const(True, "bool"),
            block=inner_block,
        )

        # Loop nest wraps the realize
        body = inner_realize
        for d in reversed(range(ndim)):
            body = tirx.For(
                loop_vars[d],
                tirx.const(0, "int32"),
                tirx.const(s[d], "int32"),
                tirx.ForKind.SERIAL,
                body,
            )

        stmts.append(body)
        offset += s[axis]

    seq = tirx.SeqStmt(stmts) if len(stmts) > 1 else stmts[0]
    params = inp_bufs + [out_buf]
    return tirx.PrimFunc(params, seq, buffer_map={}).with_attrs({
        "tirx.noalias": True,
        "s_tir": True,
    })


@mutator
class ConcatRewriter(PyExprMutator):
    def __init__(self, mod):
        super().__init__(mod)
        self._mod = mod
        self._gvar_cache = {}

    def visit_call_(self, call: relax.Call) -> relax.Expr:
        call = super().visit_call_(call)  # recurse first

        if not (hasattr(call.op, "name") and call.op.name == "relax.concat"):
            return call

        axis = int(call.attrs.axis)
        inputs = list(call.args[0])
        shapes = [list(map(int, inp.struct_info.shape)) for inp in inputs]
        dtype = inputs[0].struct_info.dtype

        out_shape = list(shapes[0])
        for s in shapes[1:]:
            out_shape[axis] += s[axis]

        # Cache key to avoid re-emitting identical funcs
        cache_key = (tuple(tuple(s) for s in shapes), axis, dtype)
        if cache_key not in self._gvar_cache:
            prim_func = make_concat_copy_func(shapes, out_shape, axis, dtype)
            gvar = self.builder_.add_func(prim_func, "concat_copy")
            self._gvar_cache[cache_key] = gvar
        else:
            gvar = self._gvar_cache[cache_key]

        out_sinfo = relax.TensorStructInfo(out_shape, dtype)
        return relax.call_tir(gvar, inputs, out_sinfo=out_sinfo)


@tvm.transform.module_pass(opt_level=0, name="ConcatToCopy")
class ConcatToCopy:
    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        rewriter = ConcatRewriter(mod)
        for g_var, func in mod.functions_items():
            if isinstance(func, relax.Function):
                func = rewriter.visit_expr(func)
                rewriter.builder_.update_func(g_var, func)
        # normalize before any further transformation, that will force the SBlocks to be marked for the tensorizer
        return normalize_mod(rewriter.builder_.get())