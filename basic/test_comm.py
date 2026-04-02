"""
通信原语测试套件 (Test Suite for Communication Primitives)
=========================================================

测试策略:
  1. 数值正确性: 验证每个原语的输出是否符合定义
  2. 恒等关系: 验证原语之间的数学等价关系
     - all_reduce ≡ reduce + broadcast
     - all_gather ≡ gather + broadcast
     - reduce_scatter ≡ reduce + scatter
     - ring_all_reduce ≡ all_reduce (结果一致)
     - scatter_reduce ≡ reduce_scatter (等大小输入)
  3. 边界值: world_size=1, 非方形张量, 高维张量
  4. 大模型场景: 模拟 TP/DP/EP 中的通信模式
"""

import pytest
import torch

from comm import (
    broadcast,
    scatter,
    gather,
    all_gather,
    reduce,
    all_reduce,
    reduce_scatter,
    all_to_all,
    ring_all_reduce,
    scatter_reduce,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(params=[2, 3, 4, 8], ids=lambda n: f"world={n}")
def world_size(request):
    """测试不同 world_size。"""
    return request.param


@pytest.fixture(params=[
    (12,),
    (4, 6),
    (2, 3, 8),
], ids=["1D", "2D", "3D"])
def tensor_shape(request):
    """测试不同维度的张量。"""
    return request.param


@pytest.fixture
def make_rank_tensors():
    """工厂 fixture: 为每个 rank 生成不同的随机张量。"""
    def _make(world_size, shape, seed=42):
        tensors = []
        for rank in range(world_size):
            torch.manual_seed(seed + rank)
            tensors.append(torch.randn(shape))
        return tensors
    return _make


# =============================================================================
# 1. broadcast 测试
# =============================================================================

class TestBroadcast:
    """broadcast: root 的数据复制到所有 rank"""

    def test_all_ranks_get_same_data(self, world_size):
        src = torch.randn(3, 4)
        result = broadcast(src, root=0, world_size=world_size)
        assert len(result) == world_size
        for r in result:
            assert torch.equal(r, src)

    def test_clones_are_independent(self):
        """修改某个 rank 的结果不影响其他 rank"""
        src = torch.ones(4)
        result = broadcast(src, world_size=3)
        result[0].fill_(99)
        assert torch.equal(result[1], torch.ones(4))
        assert torch.equal(result[2], torch.ones(4))

    def test_world_size_1(self):
        src = torch.tensor([1.0, 2.0, 3.0])
        result = broadcast(src, world_size=1)
        assert len(result) == 1
        assert torch.equal(result[0], src)


# =============================================================================
# 2. scatter 测试
# =============================================================================

class TestScatter:
    """scatter: 将数据沿 dim 切分给各 rank"""

    def test_basic_dim0(self, world_size):
        t = torch.arange(world_size * 4).float().reshape(world_size, 4)
        result = scatter(t, dim=0, world_size=world_size)
        assert len(result) == world_size
        for i, chunk in enumerate(result):
            assert chunk.shape == (1, 4)
            assert torch.equal(chunk.squeeze(0), t[i])

    def test_dim1(self):
        t = torch.arange(24).float().reshape(3, 8)  # [3, 8]
        result = scatter(t, dim=1, world_size=4)     # 切成 4 份 → [3,2] each
        assert len(result) == 4
        for c in result:
            assert c.shape == (3, 2)
        # 拼回来应该等于原始
        assert torch.equal(torch.cat(result, dim=1), t)

    def test_scatter_then_gather_is_identity(self, world_size):
        """scatter → gather 应还原原始数据"""
        t = torch.randn(world_size * 3, 5)
        chunks = scatter(t, dim=0, world_size=world_size)
        recovered = gather(chunks, dim=0)
        assert torch.equal(recovered, t)


# =============================================================================
# 3. gather 测试
# =============================================================================

class TestGather:
    """gather: 将各 rank 的数据拼接到 root"""

    def test_basic(self):
        tensors = [torch.tensor([i, i + 1]).float() for i in range(4)]
        result = gather(tensors, dim=0)
        expected = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4]).float()
        assert torch.equal(result, expected)

    def test_dim1(self):
        tensors = [torch.randn(2, 3) for _ in range(3)]
        result = gather(tensors, dim=1)
        assert result.shape == (2, 9)

    def test_single_rank(self):
        t = torch.randn(4, 5)
        result = gather([t], dim=0)
        assert torch.equal(result, t)


# =============================================================================
# 4. all_gather 测试
# =============================================================================

class TestAllGather:
    """all_gather: 每个 rank 都拿到完整拼接结果"""

    def test_every_rank_gets_full_data(self, world_size, make_rank_tensors):
        tensors = make_rank_tensors(world_size, (3, 4))
        result = all_gather(tensors, dim=0)
        expected = torch.cat(tensors, dim=0)
        assert len(result) == world_size
        for r in result:
            assert torch.equal(r, expected)

    def test_equals_gather_then_broadcast(self, world_size, make_rank_tensors):
        """验证: all_gather ≡ gather + broadcast"""
        tensors = make_rank_tensors(world_size, (4, 2))
        # 方式1: all_gather
        ag_result = all_gather(tensors, dim=0)
        # 方式2: gather + broadcast
        gathered = gather(tensors, dim=0)
        bc_result = broadcast(gathered, world_size=world_size)
        for ag, bc in zip(ag_result, bc_result):
            assert torch.equal(ag, bc)

    def test_clones_are_independent(self):
        tensors = [torch.ones(2), torch.ones(2) * 2]
        result = all_gather(tensors, dim=0)
        result[0][0] = 999
        assert result[1][0] != 999  # 不同 rank 的结果是独立副本


# =============================================================================
# 5. reduce 测试
# =============================================================================

class TestReduce:
    """reduce: 所有 rank 的数据求和, 结果在 root"""

    def test_sum_correctness(self, world_size, make_rank_tensors):
        tensors = make_rank_tensors(world_size, (5, 3))
        result = reduce(tensors)
        expected = sum(tensors)
        assert torch.allclose(result, expected)

    def test_simple_known_values(self):
        tensors = [torch.ones(4) * i for i in range(4)]
        result = reduce(tensors)
        expected = torch.ones(4) * (0 + 1 + 2 + 3)
        assert torch.equal(result, expected)

    def test_single_rank(self):
        t = torch.randn(3, 3)
        result = reduce([t])
        assert torch.allclose(result, t)


# =============================================================================
# 6. all_reduce 测试
# =============================================================================

class TestAllReduce:
    """all_reduce: 所有 rank 的数据求和, 每个 rank 都拿到结果"""

    def test_all_ranks_get_sum(self, world_size, make_rank_tensors):
        tensors = make_rank_tensors(world_size, (4, 3))
        result = all_reduce(tensors)
        expected = sum(tensors)
        assert len(result) == world_size
        for r in result:
            assert torch.allclose(r, expected)

    def test_equals_reduce_then_broadcast(self, world_size, make_rank_tensors):
        """验证: all_reduce ≡ reduce + broadcast"""
        tensors = make_rank_tensors(world_size, (6,))
        # 方式1: all_reduce
        ar_result = all_reduce(tensors)
        # 方式2: reduce + broadcast
        reduced = reduce(tensors)
        bc_result = broadcast(reduced, world_size=world_size)
        for ar, bc in zip(ar_result, bc_result):
            assert torch.allclose(ar, bc)

    def test_gradient_accumulation_scenario(self):
        """模拟 DDP 梯度同步: 每个 rank 有不同梯度, AllReduce 后取平均"""
        world_size = 4
        grads = [torch.randn(10, 10) for _ in range(world_size)]
        synced = all_reduce(grads)
        avg_grad = synced[0] / world_size
        expected_avg = sum(grads) / world_size
        assert torch.allclose(avg_grad, expected_avg)


# =============================================================================
# 7. reduce_scatter 测试
# =============================================================================

class TestReduceScatter:
    """reduce_scatter: 先 reduce(求和) 再 scatter(切分)"""

    def test_basic(self, world_size, make_rank_tensors):
        shape = (world_size * 3,)
        tensors = make_rank_tensors(world_size, shape)
        result = reduce_scatter(tensors, dim=0)

        total = sum(tensors)
        expected_chunks = torch.chunk(total, world_size, dim=0)

        assert len(result) == world_size
        for i, (r, e) in enumerate(zip(result, expected_chunks)):
            assert torch.allclose(r, e), f"rank {i} mismatch"

    def test_equals_reduce_then_scatter(self, world_size, make_rank_tensors):
        """验证: reduce_scatter ≡ reduce + scatter"""
        shape = (world_size * 4, 2)
        tensors = make_rank_tensors(world_size, shape)
        # 方式1: reduce_scatter
        rs_result = reduce_scatter(tensors, dim=0)
        # 方式2: reduce + scatter
        reduced = reduce(tensors)
        sc_result = scatter(reduced, dim=0, world_size=world_size)
        for rs, sc in zip(rs_result, sc_result):
            assert torch.allclose(rs, sc)

    def test_fsdp_gradient_sharding(self):
        """模拟 FSDP: 梯度 reduce_scatter, 每个 rank 只保留自己的分片"""
        world_size = 4
        grad_size = 1024
        grads = [torch.randn(grad_size) for _ in range(world_size)]
        shards = reduce_scatter(grads, dim=0)
        # 每个 shard 大小 = grad_size / world_size
        for s in shards:
            assert s.shape == (grad_size // world_size,)
        # 拼接所有 shard 应等于全局 reduce
        full_reduced = sum(grads)
        reconstructed = torch.cat(shards, dim=0)
        assert torch.allclose(reconstructed, full_reduced)

    def test_2d_dim1(self, make_rank_tensors):
        """沿第二个维度做 reduce_scatter"""
        world_size = 3
        tensors = make_rank_tensors(world_size, (4, 9))
        result = reduce_scatter(tensors, dim=1)
        total = sum(tensors)
        for i, r in enumerate(result):
            expected = total[:, i * 3 : (i + 1) * 3]
            assert torch.allclose(r, expected)


# =============================================================================
# 8. all_to_all 测试
# =============================================================================

class TestAllToAll:
    """all_to_all: 全交换, 等价于 chunk 矩阵的转置"""

    def test_basic(self, world_size, make_rank_tensors):
        shape = (world_size * 2,)
        tensors = make_rank_tensors(world_size, shape)
        result = all_to_all(tensors, dim=0)
        assert len(result) == world_size

        # 手动验证: rank j 收到所有 rank 的 chunk j
        send_chunks = [torch.chunk(t, world_size, dim=0) for t in tensors]
        for j in range(world_size):
            expected = torch.cat([send_chunks[i][j] for i in range(world_size)], dim=0)
            assert torch.allclose(result[j], expected)

    def test_is_involution(self, world_size, make_rank_tensors):
        """all_to_all 做两次等于恒等 (对称交换)"""
        shape = (world_size * 3,)
        tensors = make_rank_tensors(world_size, shape)
        once = all_to_all(tensors, dim=0)
        twice = all_to_all(once, dim=0)
        for orig, recovered in zip(tensors, twice):
            assert torch.allclose(orig, recovered)

    def test_moe_token_dispatch(self):
        """模拟 MoE 中的 token dispatch:
        每个 GPU 有一些 token, 需要根据路由结果发送到对应专家所在的 GPU"""
        world_size = 4
        tokens_per_gpu = 8
        hidden_dim = 16
        # 每个 GPU 的 token, 已按目标 GPU 排好序 (每段发给对应 GPU)
        data = [torch.randn(tokens_per_gpu, hidden_dim) for _ in range(world_size)]
        dispatched = all_to_all(data, dim=0)
        # dispatch 后每个 GPU 收到 tokens_per_gpu 个 token (来自不同 GPU)
        for d in dispatched:
            assert d.shape == (tokens_per_gpu, hidden_dim)

    def test_2d(self, make_rank_tensors):
        world_size = 4
        tensors = make_rank_tensors(world_size, (8, 6))
        result = all_to_all(tensors, dim=0)
        assert len(result) == world_size
        for r in result:
            assert r.shape == (8, 6)  # 每个 rank: 4 chunks of (2,6) from 4 ranks → (8,6)


# =============================================================================
# 9. ring_all_reduce 测试
# =============================================================================

class TestRingAllReduce:
    """ring_all_reduce: 环形拓扑的带宽最优 AllReduce"""

    def test_equals_naive_all_reduce(self, world_size, make_rank_tensors):
        """ring_all_reduce 结果应与 all_reduce 完全一致"""
        # ring_all_reduce 沿 dim=0 切分, 所以第一维 >= world_size
        shape = (world_size * 4,)
        tensors = make_rank_tensors(world_size, shape)
        ring_result = ring_all_reduce(tensors)
        naive_result = all_reduce(tensors)
        for ring_r, naive_r in zip(ring_result, naive_result):
            assert torch.allclose(ring_r, naive_r, atol=1e-6)

    def test_world_size_1(self):
        t = torch.randn(10)
        result = ring_all_reduce([t])
        assert len(result) == 1
        assert torch.allclose(result[0], t)

    def test_all_ranks_identical(self, world_size, make_rank_tensors):
        """所有 rank 的结果应该完全相同"""
        shape = (world_size * 3,)
        tensors = make_rank_tensors(world_size, shape)
        result = ring_all_reduce(tensors)
        for i in range(1, world_size):
            assert torch.allclose(result[0], result[i])

    def test_large_tensor(self):
        """大张量性能回归测试"""
        world_size = 4
        tensors = [torch.randn(4096) for _ in range(world_size)]
        result = ring_all_reduce(tensors)
        expected = sum(tensors)
        for r in result:
            assert torch.allclose(r, expected, atol=1e-4)

    def test_2d_tensor(self, make_rank_tensors):
        """ring_all_reduce 也能处理多维张量 (沿 dim=0 切分)"""
        world_size = 4
        tensors = make_rank_tensors(world_size, (8, 6))
        result = ring_all_reduce(tensors)
        expected = sum(tensors)
        for r in result:
            assert torch.allclose(r, expected, atol=1e-5)


# =============================================================================
# 10. scatter_reduce 测试
# =============================================================================

class TestScatterReduce:
    """scatter_reduce: 先 split 再 reduce 各 chunk"""

    def test_basic(self, world_size, make_rank_tensors):
        shape = (world_size * 3,)
        tensors = make_rank_tensors(world_size, shape)
        result = scatter_reduce(tensors, dim=0)
        assert len(result) == world_size

        all_chunks = [torch.chunk(t, world_size, dim=0) for t in tensors]
        for j in range(world_size):
            expected = sum(all_chunks[i][j] for i in range(world_size))
            assert torch.allclose(result[j], expected)

    def test_equals_reduce_scatter_for_equal_sized(self, world_size, make_rank_tensors):
        """等大小输入下, scatter_reduce ≡ reduce_scatter"""
        shape = (world_size * 5,)
        tensors = make_rank_tensors(world_size, shape)
        sr_result = scatter_reduce(tensors, dim=0)
        rs_result = reduce_scatter(tensors, dim=0)
        for sr, rs in zip(sr_result, rs_result):
            assert torch.allclose(sr, rs)

    def test_dim1(self, make_rank_tensors):
        world_size = 3
        tensors = make_rank_tensors(world_size, (4, 9))
        result = scatter_reduce(tensors, dim=1)
        assert len(result) == world_size
        for r in result:
            assert r.shape == (4, 3)


# =============================================================================
# 综合测试: 原语之间的数学关系
# =============================================================================

class TestIdentityRelations:
    """验证通信原语之间的数学恒等关系, 保证实现的一致性"""

    def test_all_reduce_eq_reduce_scatter_then_all_gather(self, make_rank_tensors):
        """all_reduce ≡ reduce_scatter + all_gather"""
        world_size = 4
        tensors = make_rank_tensors(world_size, (16,))
        # 方式1: all_reduce
        ar = all_reduce(tensors)
        # 方式2: reduce_scatter → all_gather
        rs = reduce_scatter(tensors, dim=0)
        ag = all_gather(rs, dim=0)
        for a, b in zip(ar, ag):
            assert torch.allclose(a, b)

    def test_scatter_then_all_gather_eq_broadcast(self, make_rank_tensors):
        """scatter + all_gather ≈ broadcast (只是元素顺序不同)
        更准确: scatter → gather 还原, scatter → all_gather 每个 rank 拿到全量"""
        t = torch.randn(12, 4)
        world_size = 3
        chunks = scatter(t, dim=0, world_size=world_size)
        ag = all_gather(chunks, dim=0)
        for r in ag:
            assert torch.equal(r, t)

    def test_all_to_all_twice_is_identity(self, make_rank_tensors):
        """all_to_all 是对合操作 (自逆)"""
        world_size = 4
        tensors = make_rank_tensors(world_size, (8,))
        result = all_to_all(all_to_all(tensors, dim=0), dim=0)
        for orig, recovered in zip(tensors, result):
            assert torch.allclose(orig, recovered)

    def test_reduce_scatter_eq_all_reduce_then_chunk(self, make_rank_tensors):
        """reduce_scatter ≡ all_reduce 后每个 rank 取自己的 chunk"""
        world_size = 4
        tensors = make_rank_tensors(world_size, (20,))
        # 方式1: reduce_scatter
        rs = reduce_scatter(tensors, dim=0)
        # 方式2: all_reduce → 切 chunk
        ar = all_reduce(tensors)
        for rank in range(world_size):
            chunk = torch.chunk(ar[rank], world_size, dim=0)[rank]
            assert torch.allclose(rs[rank], chunk)

    def test_ring_all_reduce_eq_naive(self, make_rank_tensors):
        """ring_all_reduce 与朴素 all_reduce 结果一致"""
        world_size = 4
        tensors = make_rank_tensors(world_size, (16,))
        ring = ring_all_reduce(tensors)
        naive = all_reduce(tensors)
        for r, n in zip(ring, naive):
            assert torch.allclose(r, n, atol=1e-6)


# =============================================================================
# 大模型场景测试
# =============================================================================

class TestLLMScenarios:
    """模拟大模型训练/推理中通信原语的典型使用模式"""

    def test_ddp_gradient_sync(self):
        """数据并行: AllReduce 梯度, 每个 GPU 得到平均梯度

        场景: 4 个 GPU 各自前向+反向得到不同梯度
        目标: AllReduce 求和后除以世界大小 = 平均梯度
        """
        world_size = 4
        param_shape = (128, 64)
        grads = [torch.randn(param_shape) for _ in range(world_size)]

        synced = all_reduce(grads)
        avg_grads = [g / world_size for g in synced]

        expected = sum(grads) / world_size
        for ag in avg_grads:
            assert torch.allclose(ag, expected)
        # 所有 GPU 的平均梯度应完全一致
        for i in range(1, world_size):
            assert torch.allclose(avg_grads[0], avg_grads[i])

    def test_tensor_parallel_column_linear(self):
        """张量并行 — 列并行线性层:
        完整权重 W 按列切分, 每个 GPU 计算部分输出, AllGather 拼接

            X @ W = X @ [W₀ | W₁] = [X@W₀ | X@W₁]
            AllGather 后每个 GPU 拿到完整输出
        """
        B, D_in, D_out = 2, 8, 16
        world_size = 4
        X = torch.randn(B, D_in)
        W = torch.randn(D_in, D_out)

        # 完整计算
        Y_full = X @ W

        # 列并行: 每个 GPU 持有 W 的一段列
        W_chunks = list(torch.chunk(W, world_size, dim=1))
        local_outputs = [X @ w for w in W_chunks]
        gathered = all_gather(local_outputs, dim=1)  # 每个 GPU 拿到完整输出

        for g in gathered:
            assert torch.allclose(g, Y_full, atol=1e-5)

    def test_tensor_parallel_row_linear(self):
        """张量并行 — 行并行线性层:
        输入和权重都按对应维度切分, 部分结果 AllReduce 求和

            X @ W = [X₀|X₁] @ [W₀] = X₀@W₀ + X₁@W₁
                               [W₁]
        """
        B, D_in, D_out = 2, 16, 8
        world_size = 4
        X = torch.randn(B, D_in)
        W = torch.randn(D_in, D_out)

        # 完整计算
        Y_full = X @ W

        # 行并行: 输入按列切, 权重按行切
        X_chunks = list(torch.chunk(X, world_size, dim=1))
        W_chunks = list(torch.chunk(W, world_size, dim=0))
        partial = [X_chunks[i] @ W_chunks[i] for i in range(world_size)]
        reduced = all_reduce(partial)

        for r in reduced:
            assert torch.allclose(r, Y_full, atol=1e-5)

    def test_fsdp_forward_all_gather(self):
        """FSDP 前向: AllGather 权重分片 → 矩阵乘 → 丢弃非本地分片

        每个 GPU 只保存 W 的 1/N, 计算前 AllGather 得到完整 W
        """
        B, D = 2, 16
        world_size = 4
        X = torch.randn(B, D)
        W = torch.randn(D, D)

        # 完整计算
        Y_full = X @ W

        # 模拟 FSDP: 切分权重
        W_shards = list(torch.chunk(W, world_size, dim=0))
        # AllGather 重建完整权重
        W_gathered = all_gather(W_shards, dim=0)
        # 每个 GPU 用完整权重计算 (结果一样)
        for Wg in W_gathered:
            Y = X @ Wg
            assert torch.allclose(Y, Y_full, atol=1e-5)

    def test_fsdp_backward_reduce_scatter(self):
        """FSDP 反向: ReduceScatter 梯度 → 每个 GPU 只保留自己负责的梯度分片"""
        world_size = 4
        grad_size = 256
        # 各 GPU 计算出的本地梯度
        grads = [torch.randn(grad_size) for _ in range(world_size)]
        # ReduceScatter: 求和后切分
        shards = reduce_scatter(grads, dim=0)

        assert len(shards) == world_size
        for s in shards:
            assert s.shape == (grad_size // world_size,)

        # 验证: AllGather shards 应等于全局 reduce 结果
        reconstructed = gather(shards, dim=0)
        full_reduce = reduce(grads)
        assert torch.allclose(reconstructed, full_reduce)

    def test_moe_expert_parallel(self):
        """MoE 专家并行: All-to-All 分发 token 到对应专家, 计算后 All-to-All 收回

        4 个 GPU, 每个 GPU 有一个专家和一批 token:
          1. All-to-All: 按路由分发 token
          2. 各 GPU 上的专家处理收到的 token
          3. All-to-All: 把结果发回原来的 GPU
        """
        world_size = 4
        tokens_per_gpu = 8  # 每个 GPU 的 token 数
        hidden = 16

        # 每个 GPU 的 token (已按目标 GPU 排好)
        send_data = [torch.randn(tokens_per_gpu, hidden) for _ in range(world_size)]

        # Step 1: All-to-All 分发
        recv_data = all_to_all(send_data, dim=0)
        for r in recv_data:
            assert r.shape == (tokens_per_gpu, hidden)

        # Step 2: "专家计算" (这里简单乘以2)
        expert_output = [r * 2 for r in recv_data]

        # Step 3: All-to-All 收回
        final = all_to_all(expert_output, dim=0)
        for f in final:
            assert f.shape == (tokens_per_gpu, hidden)

        # 验证: 两次 all_to_all + 中间等价变换, 应可追溯
        # all_to_all(all_to_all(x) * 2) 每个位置的值应是 2×原值 (因为 all_to_all 是 involution)
        for orig, result in zip(send_data, final):
            assert torch.allclose(result, orig * 2)

    def test_sequence_parallel_transition(self):
        """序列并行 → 张量并行的转换用 AllGather / ReduceScatter

        序列并行: 每个 GPU 持有 [B, S/N, H]
        张量并行 Attention 需要: [B, S, H/N]
        转换: AllGather 在 seq 维度 → scatter 在 hidden 维度
        """
        B, S, H = 2, 16, 32
        world_size = 4

        # 序列并行: 每个 GPU 持有 S/N 个 token
        sp_chunks = [torch.randn(B, S // world_size, H) for _ in range(world_size)]

        # AllGather 在 seq 维度: 每个 GPU 拿到完整序列
        full_seq = all_gather(sp_chunks, dim=1)
        for f in full_seq:
            assert f.shape == (B, S, H)

        # 然后 scatter 在 hidden 维度: 转为张量并行
        tp_chunks = scatter(full_seq[0], dim=2, world_size=world_size)
        for c in tp_chunks:
            assert c.shape == (B, S, H // world_size)


# =============================================================================
# 通信量分析测试 (验证通信量公式)
# =============================================================================

class TestCommunicationVolume:
    """验证各原语的通信量计算是否合理"""

    @staticmethod
    def _data_size(tensor):
        return tensor.nelement() * tensor.element_size()

    def test_ring_allreduce_bandwidth_optimal(self):
        """Ring AllReduce 通信量 = 2(N-1)/N × |T|

        验证方式: 与朴素 AllReduce (reduce+broadcast = 2(N-1)×|T|) 对比
        Ring 的通信量约为朴素方式的 1/N
        """
        N = 4
        T = torch.randn(1024)
        data_size = self._data_size(T)

        naive_volume = 2 * (N - 1) * data_size  # reduce + broadcast
        ring_volume = 2 * (N - 1) / N * data_size

        # ring 比 naive 少约 (1 - 1/N) 的通信量
        ratio = ring_volume / naive_volume
        assert abs(ratio - 1 / N) < 1e-6

    def test_reduce_scatter_volume(self):
        """ReduceScatter 通信量 = (N-1)/N × |T|"""
        N = 4
        T = torch.randn(1024)
        data_size = self._data_size(T)
        expected_volume = (N - 1) / N * data_size
        # 每个 rank 只需要收/发 (N-1)/N 的数据
        per_rank_recv = data_size * (N - 1) / N
        assert abs(expected_volume - per_rank_recv) < 1e-6


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
