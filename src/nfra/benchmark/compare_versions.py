"""NFRA version A/B comparison (3.1 vs 3.2, and 3.2 vs 3.3 Cortex).

Runs configs on identical code (src), identical init (torch seed), and
identical batch streams. The ONLY differences are the feature toggles, so the
delta isolates the new features:

  NFRA_CORTEX=0 (default):
    nfra31   k_wta=0 ema=off surprise=off   — 3.1 feature parity
    nfra32   KWTA    EMA    SURPRISE        — 3.2 full features
  NFRA_CORTEX=1 (3.2 vs 3.3 Cortex A/B):
    nfra32   KWTA EMA SURPRISE, Cortex OFF  — 3.2 Brain block
    nfra33   same toggles,      Cortex ON   — 3.3 Cortex block (NFRA_Cortex_Block)

Env (all optional):
  NFRA_STEPS      train steps per config        (default 1000)
  NFRA_DATA       synthetic | wikitext2         (default synthetic)
  NFRA_DIM        model width                   (default 224)
  NFRA_UNIQUE     unique fractal blocks         (default 4)
  NFRA_KWTA       k-WTA fraction for config B   (default 0.5)
  NFRA_EMA        EMA decay for config B        (default 0.99)
  NFRA_SURPRISE   1 = RPE-weighted loss for B   (default 1)
  NFRA_CORTEX     1 = compare 3.2 Brain vs 3.3 Cortex (default 0)
  NFRA_CORTEX_STATE  matrix-state width N per head (default 8)
  NFRA_EXIT_REG      exit-gate compute regularizer (default 1e-3)
  NFRA_OUT        JSON output path              (default nfra_31_vs_32.json)

Default dim/unique match the arena's 5M NFRA config, which is proven to
learn (dim 192 / 600 steps stays at random loss ~ ln(vocab), so any A/B
there is meaningless).

Writes JSON (utf-8) and prints an ASCII table.
Run locally for a smoke check; run on Kaggle T4 for the real comparison.

Usage:
  python -m nfra.benchmark.compare_versions
  NFRA_STEPS=600 NFRA_DATA=wikitext2 python -m nfra.benchmark.compare_versions
  NFRA_CORTEX=1 NFRA_OUT=nfra_32_vs_33.json python -m nfra.benchmark.compare_versions
"""

import os
import json

os.environ.setdefault('NFRA_SEEDS', '1')
os.environ.setdefault('NFRA_SIZES', '5')

import torch

from .arena import (
    build_nfra, train_one, make_loaders, NFRA_DEPTH,
    SEED_LIST, DATA_SOURCE,
)
from .compare import evaluate, count_params, DEVICE, rescale_embed

OUT_JSON = os.environ.get('NFRA_OUT', 'nfra_31_vs_32.json')

VOCAB = 96 if DATA_SOURCE == 'wikitext2' else 4096


def main():
    steps = int(os.environ.get('NFRA_STEPS', '1000'))
    dim = int(os.environ.get('NFRA_DIM', '224'))
    unique = int(os.environ.get('NFRA_UNIQUE', '4'))
    k_wta = float(os.environ.get('NFRA_KWTA', '0.5'))
    ema_decay = float(os.environ.get('NFRA_EMA', '0.99'))
    surprise = os.environ.get('NFRA_SURPRISE', '1') == '1'
    cortex = os.environ.get('NFRA_CORTEX', '0') == '1'
    cortex_state = int(os.environ.get('NFRA_CORTEX_STATE', '8'))
    exit_reg = float(os.environ.get('NFRA_EXIT_REG', '1e-3'))
    eval_gap = max(25, steps // 6)

    if cortex:
        toggles = dict(k_wta=k_wta, ema_decay=ema_decay, surprise=surprise)
        runs = [('nfra32', dict(**toggles, use_cortex=False)),
                ('nfra33', dict(**toggles, use_cortex=True))]
    else:
        runs = [('nfra31', dict(k_wta=0.0, ema_decay=0.0, surprise=False,
                                use_cortex=False)),
                ('nfra32', dict(k_wta=k_wta, ema_decay=ema_decay,
                                surprise=surprise, use_cortex=False))]

    print('NFRA A/B  |  steps=%d dim=%d unique=%d depth=%d vocab=%d data=%s'
          % (steps, dim, unique, NFRA_DEPTH, VOCAB, DATA_SOURCE))
    print('config B toggles  |  k_wta=%.2f ema=%.3f surprise=%s cortex=%s'
          % (k_wta, ema_decay, surprise, cortex))
    if cortex:
        print('cortex settings   |  d_state=%d exit_reg=%.1e'
              % (cortex_state, exit_reg))

    out = {
        'steps': steps, 'dim': dim, 'unique_blocks': unique,
        'depth': NFRA_DEPTH, 'k_wta_B': k_wta, 'ema_B': ema_decay,
        'surprise_B': surprise, 'cortex': cortex,
        'cortex_state': cortex_state, 'exit_reg': exit_reg, 'runs': {},
    }

    for name, cfg in runs:
        torch.manual_seed(0)
        train_loaders, eval_loader, _ = make_loaders(size_idx=0)
        train_loader = train_loaders[SEED_LIST[0]]

        model = build_nfra(VOCAB, dim, unique, depth=NFRA_DEPTH,
                           k_wta=cfg['k_wta'], use_cortex=cfg['use_cortex'],
                           cortex_state=cortex_state,
                           exit_reg=exit_reg).to(DEVICE)
        rescale_embed(model)
        params = count_params(model)
        res = train_one(model, VOCAB, steps, train_loader, eval_loader,
                        eval_gap, ema_decay=cfg['ema_decay'],
                        surprise=cfg['surprise'], seed=SEED_LIST[0])
        final_eval = evaluate(model, eval_loader)

        traj = [(step, round(loss, 4)) for step, loss in res['eval_hist']]
        run = {
            'k_wta': cfg['k_wta'], 'ema_decay': cfg['ema_decay'],
            'surprise': cfg['surprise'], 'use_cortex': cfg['use_cortex'],
            'params': params,
            'train_loss_first': round(res['loss_hist'][0], 4),
            'train_loss_last': round(res['loss_hist'][-1], 4),
            'final_eval_loss': round(float(final_eval), 4),
            'wall_s': round(res['wall_s'], 2),
            'tok_s': round(res['tok_s'], 1),
            'ms_per_step': round(res['ms_per_step'], 2),
            'peak_mem_gb': round(res['peak_mem'], 3),
            'nan_steps': res['nan_steps'],
            'eval_trajectory': traj,
        }
        out['runs'][name] = run
        print('[%s] params=%.3fM train %.4f -> %.4f  eval=%.4f wall=%.1fs '
              'tok_s=%.0f ms/step=%.1f nan=%d'
              % (name, params / 1e6, run['train_loss_first'],
                 run['train_loss_last'], run['final_eval_loss'],
                 run['wall_s'], run['tok_s'], run['ms_per_step'],
                 run['nan_steps']))

    names = list(out['runs'])
    a, b = out['runs'][names[-2]], out['runs'][names[-1]]
    delta = a['final_eval_loss'] - b['final_eval_loss']
    verdict = ('BETTER' if delta > 0.005 else
               'WORSE' if delta < -0.005 else 'WASH')
    print('Delta final_eval (%s - %s): %+.4f  ->  %s is %s'
          % (names[-2], names[-1], delta, names[-1], verdict))

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=True, indent=2)
    print('Wrote %s' % OUT_JSON)


if __name__ == '__main__':
    main()
