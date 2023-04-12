import argparse

from experiment import Experiment, tuning_session
import os
import json
import torch
from utils import prepare_save_folder

from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    
    # Argument Parser Setup
    parser = argparse.ArgumentParser()
    
    # Meta
    parser.add_argument("--data-dir", type=str, default="../data/")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--hyper-tune", dest="hyper_tune", action="store_true")
    parser.add_argument("--small-sample", dest="small_sample", action="store_true")
    
    # Experiment Config
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("-ep","--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    parser.add_argument("--early_stop", dest="early_stop", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--es-delta", type=float, default=0.0)
    
    # TODO
    # Cyclic LR and Data aug
    # Need to edit the utils.prepare_save_folder as well
#     parser.add_argument("--cycLR") 
#     parser.add_argument("--data-aug") 
    
    args = parser.parse_args()
    
    args.save_dir = prepare_save_folder(args)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Saving the experiment config in a json file
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
    
    print(f"INFO: Computation device: {args.device}")
    print()
    
    args.small_sample = bool(args.small_sample)
    
    if args.small_sample:
        print("INFO: This is a small sample trial.")
        print()
    
    # Experiment Section
    
    
    
    if args.hyper_tune:
        # TODO
        # Debugging
        print("INFO: This is a hyperparameter tuning session.")
        print()
        args.data_dir = os.path.abspath("../data") + "/"
        
        search_space = {
            "lr": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.grid_search([32, 64]),
            "epochs": tune.grid_search([10, 20])
        }
        
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(tuning_session, args=args),
                resources={"cpu":2, "gpu":1}
            ),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                metric="accuracy",
                mode="max",
                # search_alg=OptunaSearch(),
                num_samples=2
            )
        )
        
        results = tuner.fit()
        best_result = results.get_best_result("accuracy", "max")
        
        print("="*20)
        print("Best trial config: {}".format(best_result.config))
        print("Best trial final validation loss: {}".format(
            best_result.metrics["loss"]))
        print("Best trial final validation accuracy: {}".format(
            best_result.metrics["accuracy"]))
        print("="*20)
        
        best_model_path = f"../results/ht-{args.model}-{es}p{args.patience}-dl{args.es_delta}/lr{best_result.config["lr"]}-b{best_result.config["batch_size"]}-e{best_result.config["epochs"]}/best_model.pt"
        
        test_exp = Experiment(args)
        test_exp.load_checkpoint(best_model_path)
        results = test_exp.pred_on_test()
        results.to_csv(args.save_dir+"results.csv", index=False)
        
        
    else:
        print("INFO: This is a training session.")
        print()
        exp = Experiment(args)
        exp.fit()
        results = exp.pred_on_test()
        results.to_csv(args.save_dir+"results.csv", index=False)
        
    # TODO
    # Result plotting