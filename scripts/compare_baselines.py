from baselines.ot_icnn_map import ICNNOT
from config import WRITING_ROOT
from scripts import gauss_params
import argparse
import torch
from tools.dgps import generate_gaussian_pairs
from tools.visualize import visualize_transport

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, default=None, help='torch save file containing x and y arrays or the shapes')    
    p.add_argument('--generate', action='store_true', help='Generate gaussian paired data via tools.dgps')
    p.add_argument('--out', type=str, default=WRITING_ROOT, help='output directory')
    p.add_argument('--iters', type=int, default=gauss_params.niters, help='training steps for baselines (small default)')
    p.add_argument('--nparams', type=int, default=gauss_params.model_size, help='number of parameters')
    p.add_argument('--inner_steps', type=int, default=gauss_params.inner_steps, help='inner loop steps')
    p.add_argument('--force_retrain', action='store_true', help='Force re-computation even if output file exists')
    p.add_argument('--lr', type=float, default=gauss_params.lr, help='learning rate')
    args = p.parse_args()
    iters = args.iters
    fpath = args.data_path
    force_retrain = args.force_retrain
    nparams = args.nparams
    inner_steps = args.inner_steps
    lr = args.lr
    if fpath:
        specification = torch.load(fpath)
        if args.generate:
            x, y, _ = generate_gaussian_pairs(**specification["params"])
        else:
            x, y = specification["x"], specification["y"]
    else:
        x, y, _ = generate_gaussian_pairs(**gauss_params.params)
    d = x.shape[1]
    icnnot = ICNNOT.initialize_right_architecture(d, nparams)
    losses = icnnot.fit(x, y, iters=iters, inner_steps=inner_steps, force_retrain=force_retrain)
    visualize_transport(x, y, icnnot)

# python -m scripts.compare_baselines --data_path "/home/moeen/Documents/PhD/research/iclr/gconvex/tmp/gaussian_pairs_n10_d2_55a8fe9e77129b8acc2a238260773c82e4c094e2858747b5a4c49539d2e4e0f0.npz"