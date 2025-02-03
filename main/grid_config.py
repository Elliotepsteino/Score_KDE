from sorcerun.git_utils import (
    is_dirty,
    get_repo,
    get_commit_hash,
    get_time_str,
    get_tree_hash,
)

from config import (
    data_config,
    eval_config,
    true_density_config,
    density_estimator_config,
)

repo = get_repo()
commit_hash = get_commit_hash(repo)
time_str = get_time_str()
main_tree_hash = (get_tree_hash(repo, "main"),)
dirty = is_dirty(repo)
grid_id = (f"{time_str}--{commit_hash}--dirty={dirty}",)


def make_config(n, means, covs):
    dc = data_config.copy()
    dc["n_samples"] = n
    dc["means"] = means
    dc["covs"] = covs
    true_dc = true_density_config.copy()
    true_dc["means"] = means
    true_dc["covs"] = covs

    c = {
        "density_estimator_name": "silverman",
        "density_estimator_config": density_estimator_config,
        #
        "data_config": dc,
        "eval_config": eval_config,
        "true_density_config": true_dc,
        #
        "commit_hash": commit_hash,
        "main_tree_hash": main_tree_hash,
        "time_str": time_str,
        "dirty": dirty,
        "grid_id": grid_id,
    }
    return c


ns = [100, 1000, 10000]  # , 100000]

meanss = (
    ((-2,), (2,)),
    ((-1,), (1,)),
    ((-0.1,), (0.1,)),
)
covss = (
    (((1,),), ((1,),)),
    (((0.5,),), ((0.5,),)),
    (((0.1,),), ((0.1,),)),
)
configs = [make_config(n, m, c) for n in ns for m in meanss for c in covss]
