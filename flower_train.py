# Installed modules
import argparse
from datetime import datetime, timedelta

import flwr as fl
from flwr.common import ndarrays_to_parameters
import os
import pandas as pd
import json
import hashlib
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue
from dataclasses import dataclass
from typing import List, Optional

from src.config_params import Config

# Own modules
from src import utils
from src.datasets import data_preparation
from src.flower_manager import MyManager
from src.models import model_utils
from src.flower_strategy import MyStrategy
from src.flower_client import FlowerClient
from src import baseline_validation




global conf
global ds_split
global val_ds


# TODO: I don't know how to pass parameters to this function
def client_fn(cid: str) -> fl.client.Client:
    """Prepare flower client from ID (following flower documentation)"""
    client_id = int(cid)

    client = FlowerClient(client_id, conf)
    client_train_ds = ds_split[client_id]
    client_train_ds = data_preparation.preprocess_data(client_train_ds, conf, shuffle=True)
    # client_train_ds = data_preparation.get_ds_from_np((X_split[int(cid)], Y_split[int(cid)]))
    client.load_data(client_train_ds, val_ds)
    client.init_model()
    return client.to_client()


def train(conf, conf_name):
    """Flower training simulation using global config"""
    global ds_split
    global val_ds

    
    os.makedirs(os.path.join("checkpoints/", conf["exp_id"]), mode=0o777)
    utils.save_config(conf, os.path.join("checkpoints/", conf["exp_id"], "config.yaml"))

    train_ds, val_ds, test_ds = data_preparation.load_data(conf=conf)
    if baseline_validation.enabled(conf):
        train_ds, val_ds = baseline_validation.validation_split(train_ds, conf)
    validation_dataset = val_ds
    val_ds = data_preparation.preprocess_data(val_ds, conf, shuffle=False)
    # X_val, Y_val = data_preparation.get_np_from_ds(val_ds)
    # X_train, Y_train = data_preparation.get_np_from_ds(train_ds)
    ds_split = data_preparation.split_data(
        train_ds,
        conf
    )
    if baseline_validation.enabled(conf):
        digest = baseline_validation.save_manifest(conf, ds_split, validation_dataset, test_ds)
        print("Research data manifest SHA256:", digest)

    total_length = sum([len(ds) for ds in ds_split[:conf["dataset_options"]["num_clients"]]])

    conf["len_total_data"] = total_length

    if conf["wandb"]:
        import wandb

        setup_hash = utils.hash_config(conf, ["seed","exp_id", "len_total_data"])
        conf["config_id"] = setup_hash

        if "store_id" in conf.keys():
            if conf["store_id"]:
                conf["run_id"] = conf_name.split('.')[0]
        wandb.init(
            project="spurious_FL",
            entity="predictive-analytics-lab",
            tags=["federated"],
            config=conf,
            id=conf["exp_id"],
            job_type="train",
            reinit=True
        )

    initial_model = model_utils.init_model(
        conf=conf
    )

    model_utils.print_summary(initial_model)
    ws = model_utils.get_weights(initial_model)
    if baseline_validation.enabled(conf):
        initial_hash = hashlib.sha256()
        for array in ws:
            initial_hash.update(str((array.shape, str(array.dtype))).encode())
            initial_hash.update(array.tobytes())
        with open(os.path.join("checkpoints", conf["exp_id"], "initial_weights.sha256"), "w") as stream:
            stream.write(initial_hash.hexdigest() + "\n")
    initial_parameters = ndarrays_to_parameters(
        ws
    )

    # Create FedAvg strategy
    strategy = MyStrategy(
        conf=conf,
        initial_parameters=initial_parameters,  # avoid smaller models as init
        fraction_fit=1.0,  # Sample 10% of available clients for training
        fraction_evaluate=0.000001,  # Sample 5% of available clients for evaluation
        min_fit_clients=1,  # Never sample less than 10 clients for training
        min_evaluate_clients=1,  # Never sample less than 5 clients for evaluation
        # min_available_clients=1, # Wait until at least 75 clients are available
    )

    client_manager = MyManager(conf=conf)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=conf["dataset_options"]["num_clients"],
        config=fl.server.ServerConfig(num_rounds=conf["server_opt"]["rounds"]),
        strategy=strategy,
        client_manager=client_manager,
        ray_init_args=conf["machine"]["ray_init_args"],
        client_resources=conf["machine"]["client_resources"],
    )
    if conf["wandb"]:
        save_path = os.path.join(
                "checkpoints",
                conf["exp_id"],
                "client_weights.csv"
            )
        if os.path.exists(save_path):
            df = pd.read_csv(save_path, header=None)
            client_weights = wandb.Table(dataframe=df)
            wandb.log({"client_weights": client_weights})

        save_path = os.path.join(
                "checkpoints",
                conf["exp_id"],
                "client_info.json"
            )
        if os.path.exists(save_path):
            with open(save_path, 'r') as file:
                client_info = json.load(file)
                wandb.log({"client_info":client_info})
        wandb.finish()

    # TODO there is a new, better way of returning with latest model
    model_path = os.path.join(
        "checkpoints/",
        conf["exp_id"],
        "final"
    )
    model = model_utils.init_model(conf=conf, model_path=model_path)
    if baseline_validation.enabled(conf):
        # Per-round client evaluation uses independent validation data.
        # Test data is evaluated once, after the final checkpoint is fixed.
        from src.optimizers import optim_utils
        _, validation_accuracy, validation_groups = optim_utils.evaluate(model, val_ds, conf)
        with open(os.path.join("checkpoints", conf["exp_id"], "final_validation_metrics.json"), "w") as stream:
            json.dump({"validation_accuracy": validation_accuracy, **validation_groups}, stream, indent=2)
        test_loader = data_preparation.preprocess_data(test_ds, conf, shuffle=False)
        loss, accuracy, groups = optim_utils.evaluate(model, test_loader, conf)
        final_metrics = {"test_accuracy": accuracy, **groups,
                         "test_loss_legacy_normalization": loss,
                         "note": "Fixed-final-checkpoint test; loss retains legacy normalization."}
        with open(os.path.join("checkpoints", conf["exp_id"], "final_test_metrics.json"), "w") as stream:
            json.dump(final_metrics, stream, indent=2)
        print("FINAL_TEST_METRICS", json.dumps(final_metrics))
    return model



cs = ConfigStore.instance()
cs.store(group="job", name="federated_training", node=Config)


@hydra.main(config_path="conf", config_name="federated_training", version_base=None)
def main(cfg: Config):
    global conf
    hydra_cfg = HydraConfig.get()
    conf_name = hydra_cfg.job.config_name
    conf = OmegaConf.to_container(cfg, resolve=True)
    baseline_validation.seed_driver(conf)
    base_start_date = datetime.now()
    # Get the current run number from Hydra and add it as seconds
    try:
        run_number = hydra_cfg.job.num
        start_date = base_start_date + timedelta(seconds=run_number)
    except MissingMandatoryValue:
        start_date = base_start_date 
    conf["exp_id"] = start_date.strftime("%Y%m%d-%H%M%S")
    print(conf)
    train(conf, conf_name=conf_name)


if __name__ == "__main__":
    main()
