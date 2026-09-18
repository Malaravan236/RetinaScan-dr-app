# run_cuckoo_search.py
import numpy as np
import math
import json
from cuckoo_optimizer import CuckooSearch
from train_evaluate import train_and_evaluate
import os

# Map continuous vector to hyperparameters
def decode_solution(sol_vector):
    """
    sol_vector elements are in [0,1] scaled according to bounds mapping below.
    We'll set mapping:
      x0 -> learning_rate in [1e-5, 1e-3] (log scale)
      x1 -> dropout_rate in [0.1, 0.6]
      x2 -> dense_units in [64, 512] (discrete)
      x3 -> batch_size in [8, 64] (discrete powers of two)
    """
    # sol_vector expected in actual numeric ranges defined by bounds passed to optimizer.
    lr = float(sol_vector[0])        # actual value will be between 1e-5 and 1e-3 (we'll pass real bounds)
    dropout = float(sol_vector[1])
    dense = int(round(sol_vector[2]))
    batch = int(round(sol_vector[3]))

    # map back if needed: in our design we will give real bounds so direct values should be fine
    return {
        "learning_rate": lr,
        "dropout_rate": dropout,
        "dense_units": dense,
        "batch_size": batch,
        "epochs": 3  # small for quick fitness evaluation
    }

# objective wrapper for Cuckoo: maximize validation accuracy
def make_objective():
    cache = {}  # simple cache to avoid re-training identical hyperparams

    def obj_fn(x_vector):
        hp = decode_solution(x_vector)
        key = json.dumps(hp, sort_keys=True)
        if key in cache:
            return cache[key]

        try:
            val_acc, _, _ = train_and_evaluate(hp, quick_run=True)
        except Exception as e:
            print("Training failed for hyperparams:", hp, "error:", e)
            val_acc = 0.0

        cache[key] = val_acc
        print(f"Evaluated {hp} -> val_acc: {val_acc:.4f}")
        return val_acc

    return obj_fn

if __name__ == "__main__":
    # define bounds for each dimension (we will pass real numeric ranges)
    # x0 learning_rate in [1e-5, 1e-3]
    # x1 dropout_rate in [0.1, 0.6]
    # x2 dense_units in [64, 512]
    # x3 batch_size in [8, 64]

    bounds = [
        (1e-5, 1e-3),
        (0.1, 0.6),
        (64, 512),
        (8, 64)
    ]
    dim = len(bounds)
    objective = make_objective()

    cs = CuckooSearch(obj_function=objective, dim=dim, bounds=bounds, population_size=8, pa=0.25, alpha=0.01, max_iter=12, verbose=True)
    best_vec, best_fit = cs.run()
    print("Best solution vector:", best_vec)
    print("Best fitness (val_accuracy):", best_fit)

    # decode and run a final training with more epochs
    best_hp = {
        "learning_rate": float(best_vec[0]),
        "dropout_rate": float(best_vec[1]),
        "dense_units": int(round(best_vec[2])),
        "batch_size": int(round(best_vec[3])),
        "epochs": 12
    }
    print("Final hyperparams:", best_hp)

    # final training
    final_val_acc, final_model, history = train_and_evaluate(best_hp, quick_run=False)
    print("Final val accuracy:", final_val_acc)
    # save final model
    final_model.save(os.path.join("output", "final_model.h5"))
    print("Saved final model to output/final_model.h5")
