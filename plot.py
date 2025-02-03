# %%
import sys
import os
from sorcerun.git_utils import get_commit_hash, get_repo, get_tree_hash
import numpy as np
from sorcerun.incense_utils import (
    load_filesystem_expts_by_config_keys,
    filter_by_config,
    exps_to_xarray,
)
from tqdm import tqdm
from gifify import gifify
import matplotlib.pyplot as plt

# %%
repo = get_repo()
# get path to root of repo
REPO_PATH = repo.working_dir
sys.path.append(f"{REPO_PATH}/main")


# %%
COMMIT_HASH = get_commit_hash(repo)
SUBTREE_HASH = get_tree_hash(repo, "main")
print(f"COMMIT_HASH: {COMMIT_HASH}")
print(f"SUBTREE_HASH: {SUBTREE_HASH}")
FIG_PATH = f"{REPO_PATH}/figs/{SUBTREE_HASH}"
os.makedirs(FIG_PATH, exist_ok=True)
ALL_EXPS = load_filesystem_expts_by_config_keys(
    # main_tree_hash=SUBTREE_HASH,
    runs_dir=f"{REPO_PATH}/file_storage/runs",
    # dirty=False,
)
# %%
time_str_to_exps = {}
for exp in ALL_EXPS:
    time_str = exp.config.time_str
    if time_str not in time_str_to_exps:
        time_str_to_exps[time_str] = []
    time_str_to_exps[time_str].append(exp)
sorted_time_strs = sorted(time_str_to_exps.keys())[::-1]
single_exp_time_str = None
grid_time_str = None
for time_str in sorted_time_strs:
    if len(time_str_to_exps[time_str]) == 1:
        single_exp_time_str = (
            time_str if single_exp_time_str is None else single_exp_time_str
        )
    else:
        grid_time_str = time_str if grid_time_str is None else grid_time_str

    if single_exp_time_str is not None and grid_time_str is not None:
        break

exp = time_str_to_exps.get(single_exp_time_str, [None])[0]
exps = time_str_to_exps.get(grid_time_str, [])

# %%
from grid_config import meanss, covss

# get all exps with same
plt.figure()

for means in meanss:
    for covs in covss:
        this_exps = []
        for e in exps:
            e_means = np.array(e.config.data_config.means).reshape(-1)
            i_means = np.array(means).reshape(-1)
            e_covs = np.array(e.config.data_config.covs).reshape(-1)
            i_covs = np.array(covs).reshape(-1)

            if np.allclose(e_means, i_means) and np.allclose(e_covs, i_covs):
                this_exps.append(e)
        if len(this_exps) == 0:
            continue
        # plot kl div vs n in this exps
        data = [
            (e.config.data_config.n_samples, e.metrics["kl_divergence"][0])
            for e in this_exps
        ]
        data = np.array(data).T
        # sort by n
        data = data[:, data[0].argsort()]
        # print("A", data)
        plt.plot(*data, label=f"{means}, {covs}", marker="o")
# legend outside plot
lgd = plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.xlabel("n")
plt.ylabel("kl divergence")
plt.xscale("log")
plt.yscale("log")
plt.savefig(
    f"{FIG_PATH}/kl_div_vs_n.png",
    bbox_extra_artists=(lgd,),
    bbox_inches="tight",
)
plt.show()
