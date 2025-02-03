from sorcerun.git_utils import (
    is_dirty,
    get_repo,
    get_commit_hash,
    get_time_str,
    get_tree_hash,
)

repo = get_repo()

means = [[-2], [2]]
covs = [[[1]], [[1]]]

data_config = dict(
    n_samples=1000,
    n_dim=1,
    means=means,
    covs=covs,
)

eval_config = dict(n_eval=10000)

density_estimator_name = "silverman"
density_estimator_name = "score_informed_torch"
density_estimator_config = dict()

true_density_config = dict(
    means=means,
    covs=covs,
)


config = {
    "density_estimator_name": density_estimator_name,
    #
    "density_estimator_config": density_estimator_config,
    #
    "data_config": data_config,
    #
    "eval_config": eval_config,
    #
    "true_density_config": true_density_config,
    #
    "commit_hash": get_commit_hash(repo),
    "main_tree_hash": get_tree_hash(repo, "main"),
    "time_str": get_time_str(),
    "dirty": is_dirty(get_repo()),
}
