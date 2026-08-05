import torch
from nfra import NFRAConfig, NFRAForCausalLM

torch.manual_seed(0)
cfg = NFRAConfig(
    vocab_size=96,
    hidden_size=64,
    num_layers=3,
    n_bands=16,
    dropout=0.0,
    depth_shared=True,
    unique_blocks=3,
    use_cortex=True,
    iso_gland=True,
    iso_vgate=True,
    iso_rgate=False,
    iso_phase=True,
    iso_exit=True,
    cortex_per_token_gn=True,
)
m = NFRAForCausalLM(cfg).eval()

from nfra.core.stateful import supported, stateful_equivalence, stateful_generate_metrics

print("supported:", supported(m))
ids = torch.randint(0, 96, (1, 20))
ma, mr, ok = stateful_equivalence(m, ids, 8)
print(f"equivalence: max_abs={ma:.3e} max_rel={mr:.3e} ok={ok}")

g = stateful_generate_metrics(m, 96, prompt_len=12, gen_len=8, device="cpu")
print("gen_sf:", g["gen_sf"], "sf_ok:", g["sf_ok"])

# Direct: decoded logits for a 4-token context vs full model
from nfra.core.stateful import decode_step, make_states, prefill
st = make_states(m, 1, "cpu")
prefill(m, ids[:, :3], st)
cur = ids[:, 3:4]
logits, _ = decode_step(m, cur, st)
ref = m(ids[:, :4])["logits"][:, -1, :]
print("single-step match:", (logits[:, -1].float() - ref.float()).abs().max().item())
