import torch
import sys, importlib
import os

def is_in_jupyter_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except Exception:
        return False

IN_JUPYTER = is_in_jupyter_notebook()

def clean_torch(data):
    # takes in a dictionary and replaces the values that are torch tensors with their numpy equivalents
    cleaned = {}
    for key in data:
        if isinstance(data[key], torch.Tensor):
            cleaned[key] = data[key].squeeze().detach().numpy()
    return cleaned

def linear_kernel(X,Y, dim=1):
        return ((X * Y)).sum(dim=dim)

def quadratic_cost(choice, dim=1):
    return (choice**2).sum(dim=dim)/2

def loader(paths=None, root=None):
    """Dynamically load the optim module and import its contents into the main module. Then loads files of paths, if paths is None, the root directory is searched for all files ending with sample.pt and final.pt and these are loaded. If root is None, WRITING_ROOT from config.py is used as root.
    The function returns a dictionary with keys being the directory names and values being tuples of (sample, [models])
    """
    names = {}
    if paths is None:
        if root is None:
            from config import WRITING_ROOT as root
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root)
                # parts = rel.split(os.sep)
                key = dirpath#parts[0]
                if fn.endswith("sample.pt"):
                    key = dirpath
                    names.setdefault(key, [None, []])
                    names[key][0] = full
                if fn.endswith('final.pt'):
                    key = os.path.dirname(dirpath)
                    names.setdefault(key, [None, []])
                    names[key][1].append(full)
    
    optim_mod = importlib.import_module("optim")
    for k, v in optim_mod.__dict__.items():
        if not k.startswith("_"):
            setattr(sys.modules["__main__"], k, v)
    
    results = {}
    for key, (sample_name, models_name) in names.items():
        if sample_name is None or len(models_name) == 0:
            continue
        sample = torch.load(sample_name, map_location="cpu", weights_only=False)["all_sample"]
        models = [torch.load(f, map_location="cpu", weights_only=False) for f in models_name]
        results[key] = (sample, models)
    return results


def generate_dir_and_name(prefix, *args, **kwargs):
    #generates a directory name starting with the prefix and involving the information in args and kwargs
    dir = prefix
    for arg in args:
        if isinstance(arg, str):
            dir += f"{arg},"
        else:
            dir += f"{arg.__name__}_"
    for k, v in sorted(kwargs.items()):
        if callable(v):
            name = getattr(v, "__name__", type(v).__name__)
            dir += f"{k}:{name},"
        else:
            dir += f"{k}:{v},"
    dir = dir + "/"
    os.makedirs(dir, exist_ok=True)
    return dir

def IR_constraint(mechanism, mechanism_data, kappa = 100):
    #Constraints are supposed to be positive
    #returns the indirect utilities
    v = mechanism_data["v"]
    return v
    
def model_min_constraint(mechanism, mechanism_data, kappa=100):
    #Constraints are supposed to be positive
    #returns the distance between Y and its supposed minimum
    return mechanism.Y_rest - mechanism.y_min
    
def model_max_constraint(mechanism, mechanism_data, kappa=100):
    #Constraints are supposed to be positive
    #returns the distance between Y and its supposed minimum
    return mechanism.y_max - mechanism.Y_rest

@torch.no_grad()
def greedy_neg_order(R: torch.Tensor, k: int | None = None):
    n = R.size(0)
    if k is None: k = max(n // 5, 2)

    Rz = R.clone(); Rz.fill_diagonal_(0.0)

    # start from the most negative pair
    tri = torch.triu_indices(n, n, 1)
    idx = torch.argmin(R[tri[0], tri[1]])
    sel = [tri[0, idx].item(), tri[1, idx].item()]
    rem = set(range(n)) - set(sel)

    # greedily add the item with most negative avg corr to selected
    while rem:
        cand = torch.tensor(list(rem), device=R.device)
        j = cand[torch.argmin(Rz[cand][:, sel].mean(dim=1))].item()
        sel.append(j); rem.remove(j)

    perm = torch.tensor(sel, device=R.device)
    perm_rev = torch.flip(perm, dims=[0])

    # flip if it makes the first k block more negative (avg off-diag)
    def block_mean(p):
        if k < 2: return torch.tensor(0.0, device=R.device)
        idx = p[:k]
        B = Rz.index_select(0, idx).index_select(1, idx)
        return (B.sum() / 2) / (k * (k - 1) / 2)

    if block_mean(perm) > block_mean(perm_rev):
        perm = perm_rev

    R_perm = R.index_select(0, perm).index_select(1, perm)
    return R_perm, perm