from sorcerun.sacred_utils import run_sacred_experiment
from density_estimators import DENSITY_ESTIMATORS
from data import generate_data, true_density_mix, evaluate_estimator


def adapter(config, _run):
    density_estimator_name = config["density_estimator_name"]
    density_estimator_config = config.get("density_estimator_config", {})
    data_config = config["data_config"]
    eval_config = config["eval_config"]
    true_density_config = config["true_density_config"]

    if density_estimator_name not in DENSITY_ESTIMATORS:
        raise ValueError(f"Unknown density estimator: {density_estimator_name}")

    data = generate_data(**data_config)

    density_estimator = DENSITY_ESTIMATORS[density_estimator_name]
    estimator = density_estimator(data, **density_estimator_config)

    true_density = true_density_mix(**true_density_config)

    kl = evaluate_estimator(estimator, true_density, **eval_config)
    _run.log_scalar("kl_divergence", kl)

    return 0


adapter.experiment_name = "sample_experiment"

if __name__ == "__main__":
    from config import config

    run_sacred_experiment(adapter, config)
