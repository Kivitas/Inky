import torch

from inky.config import InkyConfig
from inky.jepa import JEPAController
from inky.memory_bank import ProductKeyMemoryBank, QuantizedValueTable


def _bank(**kw):
    args = dict(hidden_size=32, sub_keys=16, key_dim=8, mem_dim=8, top_k=4, query_steps=(1, 3))
    args.update(kw)
    return ProductKeyMemoryBank(**args)


def test_pkm_route_exact_vs_bruteforce():
    torch.manual_seed(0)
    b = _bank()
    h = torch.randn(3, 7, 32)
    idx, sc = b.pkm_route(b.norm(h))
    q = b.q_proj(b.norm(h))
    q1, q2 = q.chunk(2, -1)
    s1, s2 = q1 @ b.c1.T, q2 @ b.c2.T
    full = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).reshape(3, 7, -1)
    ref_s, ref_i = full.topk(4, dim=-1)
    assert torch.allclose(sc, ref_s, atol=1e-5)
    assert torch.equal(idx.sort(-1).values, ref_i.sort(-1).values)


def test_int4_int8_roundtrip_error_bounds():
    torch.manual_seed(0)
    v = torch.randn(64, 16)
    for dt, qmax in (("int8", 127.0), ("int4", 7.0)):
        t = QuantizedValueTable(64, 16, storage_dtype=dt, scale_dtype=torch.float32)
        t.write_slots(torch.arange(64), v)
        out = t.fetch(torch.arange(64))
        bound = v.abs().amax(-1, keepdim=True) / qmax * 0.5 + 1e-6
        assert bool(((out - v).abs() <= bound * 1.001).all()), dt
    t = QuantizedValueTable(64, 16, storage_dtype="float32")
    t.write_slots(torch.arange(64), v)
    assert torch.equal(t.fetch(torch.arange(64)), v)


def test_fetch_dedup_matches_plain_gather():
    t = QuantizedValueTable(32, 8, storage_dtype="int4")
    idx = torch.randint(0, 5, (4, 40, 2))  # heavy duplication triggers the unique path
    out = t.fetch(idx)
    ref = torch.stack([t.fetch(idx[i, j]) for i in range(4) for j in range(40)]).reshape(4, 40, 2, 8)
    assert torch.allclose(out, ref)


def test_step_conditioned_gate_changes_with_step():
    b = _bank(query_steps=(1, 2))
    with torch.no_grad():
        b.step_gate_bias[2].fill_(5.0)
    h = torch.randn(1, 3, 32)
    _, i1 = b(h, step=1)
    _, i2 = b(h, step=2)
    assert i2["gate_mean"] > i1["gate_mean"]
    assert b.step_gate_bias.shape[0] >= 3


def test_pin_memory_flag_wired_and_safe_without_cuda():
    b = _bank(pin_memory=True)
    assert b.value_table.pin_memory == torch.cuda.is_available()
    cfg = InkyConfig(enable_memory_bank=True, pkm_pin_memory=True, jepa_alpha_cos=0.9, pkm_sub_keys=8, pkm_key_dim=8, pkm_mem_dim=8, pkm_top_k=2)
    assert ProductKeyMemoryBank(cfg).value_table.pin_memory == torch.cuda.is_available()
    assert JEPAController(cfg).critic.alpha_cos == 0.9


def test_jepa_loss_decreases_and_no_collapse():
    torch.manual_seed(0)
    c = JEPAController(hidden_size=32, latent_dim=16, num_routed_experts=4)
    opt = torch.optim.Adam(c.parameters(), lr=3e-3)
    base = torch.randn(4, 12, 32)
    losses = []
    for _ in range(80):
        h = base + 0.05 * torch.randn_like(base)
        sig = c(h, step=1)
        loss = c.jepa_loss([sig.z_pred], [h.roll(-1, 1)])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert sum(losses[-5:]) < sum(losses[:5])
    z = c.encoder.encode_state(base).reshape(-1, 16)
    assert z.std(0).min() > 0.05  # not collapsed


def test_vicreg_penalizes_collapse():
    c = JEPAController(hidden_size=8, latent_dim=8, num_routed_experts=2)
    var_c, _ = c._vicreg_terms(torch.ones(4, 6, 8))
    var_n, _ = c._vicreg_terms(torch.randn(4, 6, 8) * 2)
    assert var_c > var_n


def test_gradient_flows_to_all_heads():
    torch.manual_seed(0)
    c = JEPAController(hidden_size=32, latent_dim=16, num_routed_experts=4).train()
    h = torch.randn(2, 6, 32, requires_grad=True)
    s = c(h, step=2)
    loss = (s.moe_bias.sum() + s.p_halt.sum() + s.fetch_gate.sum() + s.energy.sum()
            + c.jepa_loss([s.z_pred], [h]))
    loss.backward()
    missing = [n for n, p in c.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing, missing
    assert c.uncertainty_head.weight.grad.abs().sum() > 0
    assert c.halt_head.weight.grad.abs().sum() > 0


def test_footprint_checks_active_ram_and_bank():
    c = JEPAController(hidden_size=32, latent_dim=16, num_routed_experts=4)
    f = c.memory_footprint_bytes(bits=4)
    assert f["within_mobile_ceiling"] == 1.0 and f["active_ram_bytes"] < 1.3e9
    f16 = c.memory_footprint_bytes(bits=16)  # 50M core alone is 100MB; total blows the ceiling
    assert f16["within_total_ceiling"] == 0.0 and f16["within_mobile_ceiling"] == 1.0
    fb = c.memory_footprint_bytes(bits=4, memory_bank=_bank())
    assert fb["mmap_bank_bytes"] == _bank().value_table.storage_bytes
