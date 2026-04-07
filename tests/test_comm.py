"""Unit tests for communication primitives (comm/comm.py).

Covers all 10 primitives:
    broadcast, scatter, gather, all_gather, reduce, all_reduce,
    reduce_scatter, all_to_all, ring_all_reduce, scatter_reduce
Plus autograd wrappers and cross-primitive consistency checks.
"""

import pytest
import torch

from comm.comm import (
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
    all_reduce_autograd,
    reduce_scatter_autograd,
    all_gather_autograd,
)


# ---------------------------------------------------------------
# 1. broadcast
# ---------------------------------------------------------------

class TestBroadcast:
    def test_basic(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        results = broadcast(t, root=0, world_size=4)
        assert len(results) == 4
        for r in results:
            assert torch.allclose(r, t)

    def test_single_rank(self):
        t = torch.randn(3, 4)
        results = broadcast(t, root=0, world_size=1)
        assert len(results) == 1
        assert torch.allclose(results[0], t)

    def test_cloned(self):
        """Each rank gets an independent copy."""
        t = torch.tensor([1.0])
        results = broadcast(t, root=0, world_size=3)
        results[0].fill_(99.0)
        assert results[1].item() == 1.0  # unaffected


# ---------------------------------------------------------------
# 2. scatter
# ---------------------------------------------------------------

class TestScatter:
    def test_dim0(self):
        t = torch.arange(12).reshape(4, 3).float()
        chunks = scatter(t, dim=0, world_size=2)
        assert len(chunks) == 2
        assert torch.allclose(chunks[0], t[:2])
        assert torch.allclose(chunks[1], t[2:])

    def test_dim1(self):
        t = torch.arange(12).reshape(3, 4).float()
        chunks = scatter(t, dim=1, world_size=2)
        assert torch.allclose(chunks[0], t[:, :2])
        assert torch.allclose(chunks[1], t[:, 2:])

    def test_four_way_split(self):
        t = torch.arange(16).reshape(4, 4).float()
        chunks = scatter(t, dim=0, world_size=4)
        assert len(chunks) == 4
        for i, c in enumerate(chunks):
            assert torch.allclose(c, t[i:i+1])


# ---------------------------------------------------------------
# 3. gather
# ---------------------------------------------------------------

class TestGather:
    def test_dim0(self):
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 4.0]])
        result = gather([a, b], dim=0)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.allclose(result, expected)

    def test_dim1(self):
        a = torch.tensor([[1.0], [2.0]])
        b = torch.tensor([[3.0], [4.0]])
        result = gather([a, b], dim=1)
        expected = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        assert torch.allclose(result, expected)

    def test_returns_single_tensor(self):
        tensors = [torch.randn(2, 4) for _ in range(3)]
        result = gather(tensors, dim=0)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (6, 4)

    def test_inverse_of_scatter(self):
        t = torch.randn(8, 4)
        chunks = scatter(t, dim=0, world_size=4)
        reconstructed = gather(chunks, dim=0)
        assert torch.allclose(reconstructed, t)


# ---------------------------------------------------------------
# 4. all_gather
# ---------------------------------------------------------------

class TestAllGather:
    def test_dim0(self):
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 4.0]])
        results = all_gather([a, b], dim=0)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        for r in results:
            assert torch.allclose(r, expected)

    def test_dim1(self):
        a = torch.tensor([[1.0], [2.0]])
        b = torch.tensor([[3.0], [4.0]])
        results = all_gather([a, b], dim=1)
        expected = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        for r in results:
            assert torch.allclose(r, expected)

    def test_3d_seq_dim(self):
        world_size = 4
        B, S_local, D = 2, 8, 16
        chunks = [torch.randn(B, S_local, D) for _ in range(world_size)]
        results = all_gather(chunks, dim=1)
        expected = torch.cat(chunks, dim=1)
        for r in results:
            assert r.shape == (B, S_local * world_size, D)
            assert torch.allclose(r, expected)

    def test_every_rank_gets_same(self):
        tensors = [torch.randn(3) for _ in range(4)]
        results = all_gather(tensors, dim=0)
        for i in range(1, 4):
            assert torch.allclose(results[0], results[i])


# ---------------------------------------------------------------
# 5. reduce
# ---------------------------------------------------------------

class TestReduce:
    def test_two_ranks(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        result = reduce([a, b])
        assert torch.allclose(result, a + b)

    def test_four_ranks(self):
        tensors = [torch.randn(3, 4) for _ in range(4)]
        result = reduce(tensors)
        expected = sum(tensors)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_single_rank(self):
        t = torch.randn(5)
        result = reduce([t])
        assert torch.allclose(result, t)

    def test_returns_single_tensor(self):
        result = reduce([torch.ones(3), torch.ones(3)])
        assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------
# 6. all_reduce
# ---------------------------------------------------------------

class TestAllReduce:
    def test_two_ranks(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        results = all_reduce([a, b])
        expected = a + b
        for r in results:
            assert torch.allclose(r, expected)

    def test_four_ranks(self):
        tensors = [torch.randn(3, 4) for _ in range(4)]
        results = all_reduce(tensors)
        expected = sum(tensors)
        for r in results:
            assert torch.allclose(r, expected, atol=1e-6)

    def test_single_rank(self):
        t = torch.randn(5)
        results = all_reduce([t])
        assert torch.allclose(results[0], t)

    def test_all_ranks_identical(self):
        tensors = [torch.randn(4, 4) for _ in range(8)]
        results = all_reduce(tensors)
        for i in range(1, 8):
            assert torch.allclose(results[0], results[i])


# ---------------------------------------------------------------
# 7. reduce_scatter
# ---------------------------------------------------------------

class TestReduceScatter:
    def test_basic(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = torch.tensor([5.0, 6.0, 7.0, 8.0])
        results = reduce_scatter([a, b], dim=0)
        total = a + b  # [6, 8, 10, 12]
        assert len(results) == 2
        assert torch.allclose(results[0], total[:2])
        assert torch.allclose(results[1], total[2:])

    def test_3d(self):
        world_size = 2
        B, S, D = 2, 8, 4
        tensors = [torch.randn(B, S, D) for _ in range(world_size)]
        results = reduce_scatter(tensors, dim=1)
        total = sum(tensors)
        for i, r in enumerate(results):
            assert r.shape == (B, S // world_size, D)
            expected = total[:, i * (S // world_size):(i + 1) * (S // world_size), :]
            assert torch.allclose(r, expected, atol=1e-6)

    def test_four_ranks(self):
        tensors = [torch.randn(8) for _ in range(4)]
        results = reduce_scatter(tensors, dim=0)
        total = sum(tensors)
        assert len(results) == 4
        for i, r in enumerate(results):
            assert torch.allclose(r, total[i*2:(i+1)*2], atol=1e-6)


# ---------------------------------------------------------------
# 8. all_to_all
# ---------------------------------------------------------------

class TestAllToAll:
    def test_basic_2_ranks(self):
        t0 = torch.tensor([0.0, 1.0, 2.0, 3.0])
        t1 = torch.tensor([4.0, 5.0, 6.0, 7.0])
        results = all_to_all([t0, t1], dim=0)
        # rank 0 gets chunk0 from everyone: [0,1] from r0, [4,5] from r1
        assert torch.allclose(results[0], torch.tensor([0.0, 1.0, 4.0, 5.0]))
        # rank 1 gets chunk1 from everyone: [2,3] from r0, [6,7] from r1
        assert torch.allclose(results[1], torch.tensor([2.0, 3.0, 6.0, 7.0]))

    def test_2d_dim0(self):
        t0 = torch.arange(12).reshape(4, 3).float()
        t1 = (torch.arange(12).reshape(4, 3).float() + 100)
        results = all_to_all([t0, t1], dim=0)
        assert results[0].shape == (4, 3)
        expected_0 = torch.cat([t0[:2], t1[:2]], dim=0)
        assert torch.allclose(results[0], expected_0)

    def test_identity_single_rank(self):
        t = torch.randn(8)
        results = all_to_all([t], dim=0)
        assert len(results) == 1
        assert torch.allclose(results[0], t)

    def test_four_ranks(self):
        world_size = 4
        tensors = [torch.arange(8).float() + i * 10 for i in range(world_size)]
        results = all_to_all(tensors, dim=0)
        assert len(results) == world_size
        # total data preserved
        total_in = torch.cat(tensors, dim=0)
        total_out = torch.cat(results, dim=0)
        assert torch.allclose(total_in.sort()[0], total_out.sort()[0])


# ---------------------------------------------------------------
# 9. ring_all_reduce
# ---------------------------------------------------------------

class TestRingAllReduce:
    def test_matches_all_reduce_2_ranks(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = torch.tensor([5.0, 6.0, 7.0, 8.0])
        ring_results = ring_all_reduce([a, b])
        naive_results = all_reduce([a, b])
        for rr, nr in zip(ring_results, naive_results):
            assert torch.allclose(rr, nr, atol=1e-6)

    def test_matches_all_reduce_4_ranks(self):
        tensors = [torch.randn(16) for _ in range(4)]
        ring_results = ring_all_reduce(tensors)
        naive_results = all_reduce(tensors)
        for rr, nr in zip(ring_results, naive_results):
            assert torch.allclose(rr, nr, atol=1e-5)

    def test_single_rank(self):
        t = torch.randn(8)
        results = ring_all_reduce([t])
        assert torch.allclose(results[0], t)

    def test_all_ranks_identical(self):
        tensors = [torch.randn(12) for _ in range(3)]
        results = ring_all_reduce(tensors)
        for i in range(1, 3):
            assert torch.allclose(results[0], results[i], atol=1e-6)

    def test_large(self):
        tensors = [torch.randn(256) for _ in range(8)]
        ring_results = ring_all_reduce(tensors)
        expected = sum(tensors)
        for r in ring_results:
            assert torch.allclose(r, expected, atol=1e-4)


# ---------------------------------------------------------------
# 10. scatter_reduce
# ---------------------------------------------------------------

class TestScatterReduce:
    def test_basic(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = torch.tensor([5.0, 6.0, 7.0, 8.0])
        results = scatter_reduce([a, b], dim=0)
        assert len(results) == 2
        assert torch.allclose(results[0], torch.tensor([6.0, 8.0]))
        assert torch.allclose(results[1], torch.tensor([10.0, 12.0]))

    def test_matches_reduce_scatter(self):
        """For equal-sized inputs, scatter_reduce == reduce_scatter."""
        tensors = [torch.randn(8) for _ in range(4)]
        sr = scatter_reduce(tensors, dim=0)
        rs = reduce_scatter(tensors, dim=0)
        for a, b in zip(sr, rs):
            assert torch.allclose(a, b, atol=1e-6)

    def test_2d(self):
        world_size = 3
        tensors = [torch.randn(6, 4) for _ in range(world_size)]
        results = scatter_reduce(tensors, dim=0)
        assert len(results) == world_size
        for r in results:
            assert r.shape == (2, 4)

    def test_four_ranks(self):
        tensors = [torch.randn(12) for _ in range(4)]
        results = scatter_reduce(tensors, dim=0)
        total = sum(tensors)
        chunks = torch.chunk(total, 4, dim=0)
        for r, c in zip(results, chunks):
            assert torch.allclose(r, c, atol=1e-6)


# ---------------------------------------------------------------
# autograd wrappers
# ---------------------------------------------------------------

class TestAutograd:
    def test_all_reduce_autograd_forward(self):
        x = torch.randn(4, requires_grad=True)
        y = all_reduce_autograd(x)
        assert torch.allclose(y, x)

    def test_all_gather_autograd_roundtrip(self):
        world_size = 2
        x = torch.randn(2, 4, requires_grad=True)
        gathered = all_gather_autograd(x, world_size=world_size, rank=0, dim=0)
        assert gathered.shape == (4, 4)

    def test_reduce_scatter_autograd_shape(self):
        world_size = 2
        x = torch.randn(4, 8)
        out = reduce_scatter_autograd(x, world_size=world_size, rank=0, dim=0)
        assert out.shape == (2, 8)

    def test_all_reduce_backward(self):
        x = torch.randn(4, requires_grad=True)
        y = all_reduce_autograd(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_all_gather_backward(self):
        world_size = 2
        x = torch.randn(2, 4, requires_grad=True)
        gathered = all_gather_autograd(x, world_size=world_size, rank=0, dim=0)
        loss = gathered.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 4)


# ---------------------------------------------------------------
# Cross-primitive consistency checks
# ---------------------------------------------------------------

class TestCrossPrimitive:
    def test_all_reduce_equals_reduce_then_broadcast(self):
        tensors = [torch.randn(8) for _ in range(4)]
        ar = all_reduce(tensors)
        r = reduce(tensors)
        br = broadcast(r, root=0, world_size=4)
        for a, b in zip(ar, br):
            assert torch.allclose(a, b, atol=1e-6)

    def test_reduce_scatter_equals_reduce_then_scatter(self):
        tensors = [torch.randn(8) for _ in range(4)]
        rs = reduce_scatter(tensors, dim=0)
        r = reduce(tensors)
        sc = scatter(r, dim=0, world_size=4)
        for a, b in zip(rs, sc):
            assert torch.allclose(a, b, atol=1e-6)

    def test_all_gather_equals_gather_then_broadcast(self):
        tensors = [torch.randn(3) for _ in range(4)]
        ag = all_gather(tensors, dim=0)
        g = gather(tensors, dim=0)
        br = broadcast(g, root=0, world_size=4)
        for a, b in zip(ag, br):
            assert torch.allclose(a, b, atol=1e-6)

    def test_scatter_gather_roundtrip(self):
        t = torch.randn(12, 4)
        chunks = scatter(t, dim=0, world_size=3)
        reconstructed = gather(chunks, dim=0)
        assert torch.allclose(reconstructed, t)

    def test_ring_all_reduce_matches_all_reduce(self):
        tensors = [torch.randn(24) for _ in range(6)]
        ar = all_reduce(tensors)
        rar = ring_all_reduce(tensors)
        for a, b in zip(ar, rar):
            assert torch.allclose(a, b, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
