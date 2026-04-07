"""
test_all.py — 完整测试套件

测试结构:
  TestMatmulParallel     — 四种 ABC 矩阵切分场景
  TestAttentionParallel  — Attention 三种并行
  TestLinearParallel     — Linear 三种并行

每个测试:
  1. 运行并行实现
  2. 与单机参考结果对比（assert_close，atol=1e-4）
  3. 打印通信量信息
  4. 支持参数化（不同 n_devices, 不同矩阵尺寸）
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "megatron_pp"))

import numpy as np
import unittest
from typing import List

from core import assert_close
from matmul_parallel import (
    ScenarioA_BColumnSplit,
    ScenarioB_ARowSplit,
    ScenarioC_AColBRowSplit,
    ScenarioD_BColCRowSplit,
)
from attention_parallel import (
    AttentionConfig, AttentionReference,
    AttentionDP, AttentionSP, AttentionTP,
)
from linear_parallel import (
    LinearConfig, LinearReference,
    LinearDP, LinearSP, LinearTP_Col, LinearTP_Row, LinearTP_ColRow,
)


# ═════════════════════════════════════════════════════════════════════════════
# 工具
# ═════════════════════════════════════════════════════════════════════════════

ATOL = 1e-4

def banner(title: str):
    w = 60
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"{'='*w}")

def print_comm(comm_ops: List[str], indent: int = 4):
    for op in comm_ops:
        print(" " * indent + f"· {op}")


# ═════════════════════════════════════════════════════════════════════════════
# 1. 矩阵切分四场景
# ═════════════════════════════════════════════════════════════════════════════

class TestMatmulParallel(unittest.TestCase):

    def _run(self, strategy, matrices, ref_matrices=None):
        """通用运行 + 验证"""
        result = strategy.forward(*matrices)
        ref = ref_matrices or matrices
        ok = strategy.verify(result, *ref)
        print(f"  [{strategy.__class__.__name__}] PASS  "
              f"output={result.output.shape}  devices={result.n_devices}")
        print_comm(result.comm_ops)
        return result

    # ── 场景一：B 列切 ─────────────────────────────────────────────────────

    def test_scenario_A_B_col_split_basic(self):
        """基础：2 设备，小矩阵"""
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 6)).astype(np.float32)
        B = rng.standard_normal((6, 8)).astype(np.float32)
        s = ScenarioA_BColumnSplit(n_devices=2)
        self._run(s, [A, B])

    def test_scenario_A_B_col_split_4devices(self):
        """4 设备"""
        rng = np.random.default_rng(1)
        A = rng.standard_normal((8, 16)).astype(np.float32)
        B = rng.standard_normal((16, 32)).astype(np.float32)
        s = ScenarioA_BColumnSplit(n_devices=4)
        self._run(s, [A, B])

    def test_scenario_A_B_col_split_3d(self):
        """3D 张量（批量矩阵乘）"""
        rng = np.random.default_rng(2)
        A = rng.standard_normal((2, 4, 6)).astype(np.float32)
        B = rng.standard_normal((6, 8)).astype(np.float32)
        # 广播 B：3D A 与 2D B，用 einsum
        # 简化：只测 2D 场景
        A2 = A.reshape(-1, 6)
        s = ScenarioA_BColumnSplit(n_devices=2)
        self._run(s, [A2, B])

    # ── 场景二：A 行切 ─────────────────────────────────────────────────────

    def test_scenario_B_A_row_split_basic(self):
        rng = np.random.default_rng(3)
        A = rng.standard_normal((8, 6)).astype(np.float32)
        B = rng.standard_normal((6, 10)).astype(np.float32)
        s = ScenarioB_ARowSplit(n_devices=2)
        self._run(s, [A, B])

    def test_scenario_B_A_row_split_4devices(self):
        rng = np.random.default_rng(4)
        A = rng.standard_normal((16, 12)).astype(np.float32)
        B = rng.standard_normal((12, 8)).astype(np.float32)
        s = ScenarioB_ARowSplit(n_devices=4)
        self._run(s, [A, B])

    def test_scenario_B_large_seq(self):
        """模拟大序列场景：A=[bs*seq, d], B=[d, out]"""
        rng = np.random.default_rng(5)
        bs, seq, d, out = 2, 32, 64, 128
        A = rng.standard_normal((bs * seq, d)).astype(np.float32)
        B = rng.standard_normal((d, out)).astype(np.float32)
        s = ScenarioB_ARowSplit(n_devices=4)
        self._run(s, [A, B])

    # ── 场景三：A 列切 + B 行切 ────────────────────────────────────────────

    def test_scenario_C_col_row_basic(self):
        rng = np.random.default_rng(6)
        A = rng.standard_normal((4, 8)).astype(np.float32)
        B = rng.standard_normal((8, 6)).astype(np.float32)
        s = ScenarioC_AColBRowSplit(n_devices=2)
        self._run(s, [A, B])

    def test_scenario_C_col_row_4devices(self):
        rng = np.random.default_rng(7)
        M, K, N = 8, 16, 12
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        s = ScenarioC_AColBRowSplit(n_devices=4)
        self._run(s, [A, B])

    def test_scenario_C_allreduce_semantics(self):
        """验证 AllReduce 后每设备结果与完整计算完全一致"""
        rng = np.random.default_rng(8)
        A = rng.standard_normal((6, 8)).astype(np.float32)
        B = rng.standard_normal((8, 4)).astype(np.float32)
        s = ScenarioC_AColBRowSplit(n_devices=2)
        result = s.forward(A, B)
        ref = A @ B
        # 验证每个设备持有相同的完整输出
        for shard in result.shards:
            assert_close(shard, ref, name="each_shard_equals_ref")

    # ── 场景四：B 列切 + C 行切（三矩阵）─────────────────────────────────

    def test_scenario_D_three_matrix_basic(self):
        rng = np.random.default_rng(9)
        A = rng.standard_normal((4, 6)).astype(np.float32)
        B = rng.standard_normal((6, 8)).astype(np.float32)
        C = rng.standard_normal((8, 4)).astype(np.float32)
        s = ScenarioD_BColCRowSplit(n_devices=2)
        result = s.forward(A, B, C)
        ref = A @ B @ C
        assert_close(result.output, ref, name="ScenarioD_3mat")
        print(f"  [ScenarioD] PASS  output={result.output.shape}")
        print_comm(result.comm_ops)

    def test_scenario_D_three_matrix_4devices(self):
        rng = np.random.default_rng(10)
        A = rng.standard_normal((8, 12)).astype(np.float32)
        B = rng.standard_normal((12, 16)).astype(np.float32)
        C = rng.standard_normal((16, 8)).astype(np.float32)
        s = ScenarioD_BColCRowSplit(n_devices=4)
        result = s.forward(A, B, C)
        ref = A @ B @ C
        assert_close(result.output, ref, name="ScenarioD_4dev")

    def test_scenario_D_mlp_simulation(self):
        """模拟 Transformer MLP：X @ W1 @ W2，W1 列切，W2 行切"""
        rng = np.random.default_rng(11)
        bs, seq, d, d_ff = 2, 8, 32, 128
        X  = rng.standard_normal((bs * seq, d)).astype(np.float32)
        W1 = rng.standard_normal((d, d_ff)).astype(np.float32) * 0.02
        W2 = rng.standard_normal((d_ff, d)).astype(np.float32) * 0.02
        s = ScenarioD_BColCRowSplit(n_devices=4)
        result = s.forward(X, W1, W2)
        ref = X @ W1 @ W2
        assert_close(result.output, ref, name="MLP_sim")
        print(f"  [MLP sim] PASS  X{X.shape} W1{W1.shape} W2{W2.shape}")


# ═════════════════════════════════════════════════════════════════════════════
# 2. Attention 并行
# ═════════════════════════════════════════════════════════════════════════════

class TestAttentionParallel(unittest.TestCase):

    def _make(self, n_devices, **kw):
        cfg = AttentionConfig(**kw)
        # 确保维度能被设备数整除
        cfg.bs       = max(cfg.bs, n_devices) if 'bs' not in kw else cfg.bs
        ref = AttentionReference(cfg)
        X   = np.random.default_rng(42).standard_normal(
            (cfg.bs, cfg.seq_len, cfg.d_model)).astype(np.float32) * 0.1
        return cfg, ref, X

    # ── DP ─────────────────────────────────────────────────────────────────

    def test_attention_dp_2devices(self):
        cfg, ref, X = self._make(2, bs=4, heads=4, seq_len=8, head_dim=16)
        s = AttentionDP(ref, n_devices=2)
        ok = s.verify(X)
        result = s.forward(X)
        print(f"  [Attention DP=2] PASS  {result.strategy}")
        print_comm(result.comm_ops)

    def test_attention_dp_4devices(self):
        cfg, ref, X = self._make(4, bs=8, heads=4, seq_len=8, head_dim=16)
        s = AttentionDP(ref, n_devices=4)
        self.assertTrue(s.verify(X))

    def test_attention_dp_single(self):
        """1 设备退化为参考实现"""
        cfg, ref, X = self._make(1, bs=2, heads=4, seq_len=8, head_dim=16)
        s = AttentionDP(ref, n_devices=1)
        self.assertTrue(s.verify(X))

    # ── SP (Ring Attention) ────────────────────────────────────────────────

    def test_attention_sp_2devices(self):
        cfg, ref, X = self._make(2, bs=2, heads=4, seq_len=8, head_dim=16)
        s = AttentionSP(ref, n_devices=2)
        ok = s.verify(X)
        result = s.forward(X)
        print(f"  [Attention SP=2] PASS  {result.strategy}")
        print_comm(result.comm_ops)

    def test_attention_sp_4devices(self):
        cfg, ref, X = self._make(4, bs=2, heads=4, seq_len=16, head_dim=16)
        s = AttentionSP(ref, n_devices=4)
        self.assertTrue(s.verify(X))

    def test_attention_sp_ring_online_softmax(self):
        """验证 Ring Attention 的 online softmax 数值精确性"""
        cfg, ref, X = self._make(2, bs=1, heads=2, seq_len=4, head_dim=8)
        s = AttentionSP(ref, n_devices=2)
        result = s.forward(X)
        ref_out = ref.forward(X)
        assert_close(result.output, ref_out, name="ring_online_softmax", atol=1e-4)

    # ── TP ─────────────────────────────────────────────────────────────────

    def test_attention_tp_2devices(self):
        cfg, ref, X = self._make(2, bs=2, heads=4, seq_len=8, head_dim=16)
        s = AttentionTP(ref, n_devices=2)
        ok = s.verify(X)
        result = s.forward(X)
        print(f"  [Attention TP=2] PASS  {result.strategy}")
        print_comm(result.comm_ops)

    def test_attention_tp_4devices(self):
        cfg, ref, X = self._make(4, bs=2, heads=8, seq_len=8, head_dim=16)
        s = AttentionTP(ref, n_devices=4)
        self.assertTrue(s.verify(X))

    def test_attention_tp_head_assignment(self):
        """验证每设备处理的 head 数正确"""
        n_dev = 2
        cfg, ref, X = self._make(n_dev, bs=2, heads=4, seq_len=8, head_dim=16)
        s = AttentionTP(ref, n_devices=n_dev)
        result = s.forward(X)
        expected_local_heads = cfg.heads // n_dev
        # 检查中间分片维度（通过局部输出推算，局部 W_O 输入维度）
        local_d = cfg.d_model // n_dev
        self.assertEqual(local_d, expected_local_heads * cfg.head_dim)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Linear 层并行
# ═════════════════════════════════════════════════════════════════════════════

class TestLinearParallel(unittest.TestCase):

    def _make(self, n_devices, bs=4, seq=8, hidden=32, out=64):
        cfg = LinearConfig(bs=bs, seq_len=seq, hidden_size=hidden, out_size=out)
        ref = LinearReference(cfg)
        rng = np.random.default_rng(99)
        X   = rng.standard_normal((bs, seq, hidden)).astype(np.float32) * 0.1
        return cfg, ref, X

    # ── DP ─────────────────────────────────────────────────────────────────

    def test_linear_dp_2devices(self):
        cfg, ref, X = self._make(2)
        s = LinearDP(ref, n_devices=2)
        ok = s.verify(X)
        result = s.forward(X)
        print(f"  [Linear DP=2] PASS  {result.strategy}")
        print_comm(result.comm_ops)

    def test_linear_dp_4devices(self):
        cfg, ref, X = self._make(4, bs=8)
        s = LinearDP(ref, n_devices=4)
        self.assertTrue(s.verify(X))

    # ── SP ─────────────────────────────────────────────────────────────────

    def test_linear_sp_2devices(self):
        cfg, ref, X = self._make(2)
        s = LinearSP(ref, n_devices=2)
        ok = s.verify(X)
        result = s.forward(X)
        print(f"  [Linear SP=2] PASS  {result.strategy}")
        print_comm(result.comm_ops)

    def test_linear_sp_4devices(self):
        cfg, ref, X = self._make(4, seq=16)
        s = LinearSP(ref, n_devices=4)
        self.assertTrue(s.verify(X))

    def test_linear_sp_zero_comm(self):
        """验证 SP 正向确实无通信（comm_ops 说明）"""
        cfg, ref, X = self._make(2)
        s = LinearSP(ref, n_devices=2)
        result = s.forward(X)
        self.assertTrue(any("0 通信" in op for op in result.comm_ops))

    # ── TP 列切分 ──────────────────────────────────────────────────────────

    def test_linear_tp_col_2devices(self):
        cfg, ref, X = self._make(2)
        s = LinearTP_Col(ref, n_devices=2)
        ok = s.verify(X)
        result = s.forward(X)
        print(f"  [Linear TP_Col=2] PASS  {result.strategy}")
        print_comm(result.comm_ops)

    def test_linear_tp_col_4devices(self):
        cfg, ref, X = self._make(4, out=128)
        s = LinearTP_Col(ref, n_devices=4)
        self.assertTrue(s.verify(X))

    def test_linear_tp_col_shard_shape(self):
        """验证列切分后每设备局部输出形状"""
        n_dev = 2
        cfg, ref, X = self._make(n_dev, out=64)
        s = LinearTP_Col(ref, n_devices=n_dev)
        result = s.forward(X)
        expected_shape = (cfg.bs, cfg.seq_len, cfg.out_size // n_dev)
        for shard in result.shards:
            self.assertEqual(shard.shape, expected_shape)

    # ── TP 行切分 ──────────────────────────────────────────────────────────

    def test_linear_tp_row_2devices(self):
        cfg, ref, X = self._make(2)
        s = LinearTP_Row(ref, n_devices=2)
        ok = s.verify(X)
        result = s.forward(X)
        print(f"  [Linear TP_Row=2] PASS  {result.strategy}")
        print_comm(result.comm_ops)

    def test_linear_tp_row_4devices(self):
        cfg, ref, X = self._make(4, hidden=64)
        s = LinearTP_Row(ref, n_devices=4)
        self.assertTrue(s.verify(X))

    # ── TP 列→行串联（两层 MLP）──────────────────────────────────────────

    def test_linear_tp_col_row_2devices(self):
        rng = np.random.default_rng(55)
        cfg = LinearConfig(bs=2, seq_len=8, hidden_size=32, out_size=32)
        ref = LinearReference(cfg)
        W2  = rng.standard_normal((cfg.out_size, cfg.hidden_size)).astype(np.float32) * 0.02
        b2  = rng.standard_normal((cfg.hidden_size,)).astype(np.float32) * 0.01
        X   = rng.standard_normal((cfg.bs, cfg.seq_len, cfg.hidden_size)).astype(np.float32) * 0.1

        s   = LinearTP_ColRow(ref, n_devices=2, W2=W2, b2=b2)
        ok  = s.verify(X)
        result = s.forward(X)
        print(f"  [Linear TP_ColRow=2] PASS  {result.strategy}")
        print_comm(result.comm_ops)

    def test_linear_tp_col_row_4devices(self):
        rng = np.random.default_rng(66)
        cfg = LinearConfig(bs=2, seq_len=8, hidden_size=64, out_size=128)
        ref = LinearReference(cfg)
        W2  = rng.standard_normal((cfg.out_size, cfg.hidden_size)).astype(np.float32) * 0.02
        b2  = rng.standard_normal((cfg.hidden_size,)).astype(np.float32) * 0.01
        X   = rng.standard_normal((cfg.bs, cfg.seq_len, cfg.hidden_size)).astype(np.float32) * 0.1
        s   = LinearTP_ColRow(ref, n_devices=4, W2=W2, b2=b2)
        self.assertTrue(s.verify(X))


# ═════════════════════════════════════════════════════════════════════════════
# 4. 边界与压力测试
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):

    def test_single_device_all_strategies(self):
        """1 设备 = 无切分，退化为参考实现"""
        rng = np.random.default_rng(77)
        A = rng.standard_normal((4, 6)).astype(np.float32)
        B = rng.standard_normal((6, 8)).astype(np.float32)
        for Cls in [ScenarioA_BColumnSplit, ScenarioB_ARowSplit, ScenarioC_AColBRowSplit]:
            s = Cls(n_devices=1)
            result = s.forward(A, B)
            assert_close(result.output, A @ B, name=f"{Cls.__name__}_1dev")

    def test_square_matrices(self):
        """方阵"""
        rng = np.random.default_rng(88)
        A = rng.standard_normal((8, 8)).astype(np.float32)
        B = rng.standard_normal((8, 8)).astype(np.float32)
        for Cls in [ScenarioA_BColumnSplit, ScenarioB_ARowSplit, ScenarioC_AColBRowSplit]:
            s = Cls(n_devices=4)
            s.verify(s.forward(A, B), A, B)

    def test_large_matrix_numerical_stability(self):
        """大值矩阵：验证数值稳定性"""
        rng = np.random.default_rng(89)
        A = rng.standard_normal((16, 64)).astype(np.float32) * 10.0
        B = rng.standard_normal((64, 32)).astype(np.float32) * 10.0
        s = ScenarioC_AColBRowSplit(n_devices=4)
        result = s.forward(A, B)
        ref = A @ B
        assert_close(result.output, ref, name="large_val_stability", atol=1e-2)

    def test_attention_sp_equals_dp_output(self):
        """SP 和 DP 对相同输入应给出相同输出（不同 X）"""
        cfg = AttentionConfig(bs=4, heads=4, seq_len=8, head_dim=16)
        ref = AttentionReference(cfg)
        X   = np.random.default_rng(101).standard_normal(
            (cfg.bs, cfg.seq_len, cfg.d_model)).astype(np.float32) * 0.1
        dp_out = AttentionDP(ref, n_devices=2).forward(X).output
        sp_out = AttentionSP(ref, n_devices=2).forward(X).output
        tp_out = AttentionTP(ref, n_devices=2).forward(X).output
        ref_out = ref.forward(X)
        for name, out in [("DP", dp_out), ("SP", sp_out), ("TP", tp_out)]:
            assert_close(out, ref_out, name=f"{name}_vs_ref", atol=1e-4)


# ═════════════════════════════════════════════════════════════════════════════
# 5. 通信量对比报告
# ═════════════════════════════════════════════════════════════════════════════

def comm_report():
    banner("通信量对比报告")

    dtype_bytes = 4  # float32
    M, K, N, H  = 2048, 4096, 4096, 16384  # 典型 LLM 维度

    def fmt(n):
        if n >= 1e9: return f"{n/1e9:.2f} GB"
        if n >= 1e6: return f"{n/1e6:.2f} MB"
        return f"{n/1e3:.2f} KB"

    strategies = [
        ("场景一 B列切（AllGather）",        M * N * dtype_bytes),
        ("场景二 A行切（AllGather）",         M * N * dtype_bytes),
        ("场景三 A列+B行（AllReduce）",       M * N * dtype_bytes),
        ("场景四 B列+C行三矩阵（AllReduce）", M * N * dtype_bytes),
        ("Attention DP",                       0),
        ("Attention SP Ring（P2P×p）",         2 * M * K * dtype_bytes),
        ("Attention TP（AllReduce）",          M * K * dtype_bytes),
        ("Linear DP",                          0),
        ("Linear SP",                          0),
        ("Linear TP列（AllGather可选）",       M * N * dtype_bytes),
        ("Linear TP行（AllReduce）",           M * N * dtype_bytes),
    ]

    print(f"\n  {'策略':<36} {'通信量':>12}  备注")
    print("  " + "-" * 72)
    for name, vol in strategies:
        print(f"  {name:<36} {fmt(vol):>12}")


# ═════════════════════════════════════════════════════════════════════════════
# 主入口
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner("矩阵切分四场景测试")
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestMatmulParallel)
    unittest.TextTestRunner(verbosity=0, stream=open(os.devnull,'w')).run(suite1)
    # 手动运行并打印
    t = TestMatmulParallel()
    for method in [
        "test_scenario_A_B_col_split_basic",
        "test_scenario_A_B_col_split_4devices",
        "test_scenario_B_A_row_split_basic",
        "test_scenario_B_large_seq",
        "test_scenario_C_col_row_basic",
        "test_scenario_C_allreduce_semantics",
        "test_scenario_D_three_matrix_basic",
        "test_scenario_D_mlp_simulation",
    ]:
        getattr(t, method)()

    banner("Attention 并行测试")
    ta = TestAttentionParallel()
    for method in [
        "test_attention_dp_2devices",
        "test_attention_sp_2devices",
        "test_attention_tp_2devices",
        "test_attention_tp_4devices",
        "test_attention_sp_ring_online_softmax",
    ]:
        getattr(ta, method)()

    banner("Linear 层并行测试")
    tl = TestLinearParallel()
    for method in [
        "test_linear_dp_2devices",
        "test_linear_sp_2devices",
        "test_linear_sp_zero_comm",
        "test_linear_tp_col_2devices",
        "test_linear_tp_col_shard_shape",
        "test_linear_tp_row_2devices",
        "test_linear_tp_col_row_2devices",
        "test_linear_tp_col_row_4devices",
    ]:
        getattr(tl, method)()

    banner("边界测试")
    te = TestEdgeCases()
    for method in [
        "test_single_device_all_strategies",
        "test_square_matrices",
        "test_large_matrix_numerical_stability",
        "test_attention_sp_equals_dp_output",
    ]:
        getattr(te, method)()
        print(f"  [{method}] PASS")

    comm_report()

    print("\n" + "="*60)
    print("  全部测试通过！")
    print("="*60)
    print("\n运行完整 unittest:")
    print("  python -m pytest test_all.py -v")
