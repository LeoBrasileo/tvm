/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt.h>

#include "../../schedule/analysis.h"
#include "../../schedule/transform.h"
#include "../utils.h"
#include "multi_level_tiling.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

constexpr const char* kSpatialTensorizedAttr = "meta_schedule.spatial_tensorized";

// Declared in multi_level_tiling_with_intrin.cc
ffi::Optional<s_tir::SBlockRV> TileForIntrin(s_tir::Schedule sch, s_tir::SBlockRV block,
                                             const std::string& intrin_name);

/*!
 * \brief Tensorization rule for blocks that have NO reduction axis
 */
class SpatialTilingWithIntrinNode : public MultiLevelTilingNode {
 public:
  ffi::String intrin_name;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SpatialTilingWithIntrinNode>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.SpatialTilingWithIntrin",
                                    SpatialTilingWithIntrinNode, MultiLevelTilingNode);

 protected:
  ffi::Array<s_tir::Schedule> Apply(const s_tir::Schedule& sch,
                                    const s_tir::SBlockRV& block_rv) final {
    tirx::StmtSRef block_sref = sch->GetSRef(block_rv);
    const tirx::SBlockNode* block_node = block_sref->StmtAs<tirx::SBlockNode>();
    TVM_FFI_ICHECK(block_node != nullptr);
    for (const tirx::IterVar& iter_var : block_node->iter_vars) {
      if (iter_var->iter_type == tirx::IterVarType::kCommReduce) {
        return {sch};
      }
    }

    if (GetAnn<ffi::String>(block_sref, kSpatialTensorizedAttr)) {
      return {sch};
    }

    auto desc_func = tirx::TensorIntrin::Get(intrin_name).value()->desc;
    if (!CheckAutoTensorizeApplicable(sch, block_rv, desc_func)) {
      //TVM_PY_LOG(INFO, logger) << "The workload cannot be tensorized.";
      return {sch};
    }

    Schedule sch_copy = sch->Copy();
    sch_copy->Annotate(block_rv, s_tir::attr::meta_schedule_tiling_structure, structure);
    std::vector initial_states{State(sch_copy, block_rv)};
    std::vector<State> states = ApplySubRules(initial_states);
    if (states.empty()) {
      //TVM_PY_LOG(INFO, logger) << "The workload cannot be tensorized.";
      return {sch};
    }

    ffi::Array<s_tir::Schedule> results;
    results.push_back(sch);
    for (auto&& state : states) {
      results.push_back(std::move(state->sch));
    }
    TVM_PY_LOG(INFO, logger) << "Tensorizing with " << intrin_name;
    return results;
  }

  std::vector<State> ApplySubRules(std::vector<State> states) final {
    states = SubRule(std::move(states), [&](State state) {
      if (auto block_rv = TileForIntrin(state->sch, state->block_rv, intrin_name)) {
        // Mark block as tensorized to prevent cascading
        state->sch->Annotate(state->block_rv, kSpatialTensorizedAttr, ffi::String("1"));
        TVM_PY_LOG(INFO, logger) << "Matched intrin " << intrin_name << " to block, resulting loops: "
            << state->sch->GetLoops(block_rv.value()).size();
        state->block_rv = block_rv.value();
        return std::vector<State>(1, state);
      }
      return std::vector<State>();
    });
    states = SubRule(std::move(states),
                     [&](State state) { return TileLoopNest(std::move(state)); });
    return states;
  }

  ScheduleRule Clone() const final {
    ffi::ObjectPtr<SpatialTilingWithIntrinNode> n =
        ffi::make_object<SpatialTilingWithIntrinNode>(*this);
    return ScheduleRule(n);
  }
};

ScheduleRule ScheduleRule::SpatialTilingWithIntrin(
    ffi::String intrin_name, ffi::String structure,
    ffi::Optional<int64_t> max_innermost_factor,
    ffi::Optional<ffi::Map<ffi::String, ffi::Any>> reuse_read,
    ffi::Optional<ffi::Map<ffi::String, ffi::Any>> reuse_write) {
  TVM_FFI_ICHECK(tirx::TensorIntrin::Get(intrin_name).defined())
      << "Provided tensor intrinsic " << intrin_name << " is not registered.";
  for (char c : std::string(structure)) {
    TVM_FFI_ICHECK(c == 'S')
        << "SpatialTilingWithIntrin requires an all-spatial structure (only 'S' levels), got: "
        << structure << ". Use MultiLevelTilingWithIntrin for blocks with a reduction axis.";
  }

  ffi::ObjectPtr<SpatialTilingWithIntrinNode> node =
      MultiLevelTilingInitCommon<SpatialTilingWithIntrinNode>(
          structure, /*tile_binds=*/std::nullopt, max_innermost_factor,
          /*vector_load_lens=*/std::nullopt, reuse_read, reuse_write);
  node->intrin_name = intrin_name;
  return ScheduleRule(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  SpatialTilingWithIntrinNode::RegisterReflection();
  refl::GlobalDef().def("s_tir.meta_schedule.ScheduleRuleSpatialTilingWithIntrin",
                        ScheduleRule::SpatialTilingWithIntrin);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm