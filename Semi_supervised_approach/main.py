import os
import json
import torch
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from Semi_supervised_approach.models import  train_node_classification, NodeEncoder
from Semi_supervised_approach.data import embed_texts, build_knn_graph, load_clean_dataset, add_attribute_co_membership_edges


# TARGET = "Hostility to out-group (e.g. verbal attacks, belittlement, instillment of fear, incitement to violence)"
TARGET = "Polarization/Othering"
# TARGET = "Intolerance"
# TARGET = "Topic"

SEEDS = [0, 21, 42, 84, 123]
EPOCHS = 600
LR = 1e-3
WEIGHT_DECAY = 1e-5

HYPERPARAMS = {
    "hidden_dim": 128,
    "dropout": 0.2,   
    "activation": "elu",
    "jk_mode": "lstm",
    "lr": LR
}

REUSE_SPLITS_PER_K = True

# Utils
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(d):
    if d is None:
        return
    os.makedirs(d, exist_ok=True)

def bytes_to_mb(b):
    return float(b) / (1024 ** 2)



def run_graphsage_experiment(
    df,
    target_col,
    embed_texts,
    build_knn_graph,
    add_attribute_co_membership_edges,
    NodeEncoder,
    train_node_classification,
    HYPERPARAMS,
    EPOCHS,
    WEIGHT_DECAY,
    SEEDS,
    folder_name=None,
):
    # Save dir
    save_dir = None
    if folder_name:
        save_dir = os.path.join("verification_results", folder_name)
        ensure_dir(save_dir)
        print(f"\nCreated/using directory: {save_dir}\n")
    else:
        print("\nNo folder specified - results will not be saved.\n")

    # Prepare texts and labels once
    texts = df['Text'].tolist()
    groups = df[target_col].fillna(False).tolist()
    unique_groups = sorted(set(groups))
    group2id = {g: i for i, g in enumerate(unique_groups)}
    id2group = {i: g for g, i in group2id.items()}
    labels = torch.tensor([group2id[g] for g in groups], dtype=torch.long)

    # Embeddings
    print("Embedding texts (cached for all k)...")
    t0 = time.perf_counter()
    emb = embed_texts(texts)
    embed_time = time.perf_counter() - t0

    embed_flops = None
    try:
        if hasattr(embed_texts, "estimate_flops"):
            embed_flops = int(embed_texts.estimate_flops(len(texts)))
    except Exception:
        embed_flops = None

    print(f"Embedding done — time: {embed_time:.2f}s, flops: {embed_flops}")

    # Cache k-graphs to avoid rebuilding
    knn_cache = {}

    results_by_k = {}

    for k in [5, 10, 15, 20]:
        print("\n" + "-" * 50)
        print(f"--- Running for k = {k} ---")
        print("-" * 50)

        # Build or reuse kNN graph
        if k in knn_cache:
            data = knn_cache[k]
            print(f"Using cached graph for k={k}")
        else:
            print(f"Building kNN graph for k={k} ...")
            t_graph = time.perf_counter()
            data = build_knn_graph(emb, k=k)
            # Add attribute-based edges
            
            attrs = {
                'In-Group': df['In-Group'].tolist(),
                'Out-group': df['Out-group'].tolist()
            }
            data = add_attribute_co_membership_edges(data, attrs, max_edges_per_node=20, embeddings=emb)
            
            
            knn_cache[k] = data
            print(f"Built graph in {time.perf_counter() - t_graph:.2f}s")

        # Basic graph stats
        N = int(getattr(data, "num_nodes", data.x.shape[0]))
        E = int(data.edge_index.shape[1])
        avg_deg = E / N
        in_dim = data.x.shape[1]
        num_classes = len(unique_groups)
        print(f"Nodes: {N}, Edges: {E}, Avg deg: {avg_deg:.2f}, Input dim: {in_dim}, Classes: {num_classes}")

        # Precompute train/val/test split indices once per k (optional reuse)
        if REUSE_SPLITS_PER_K:
            idx = np.arange(N)
            np.random.shuffle(idx)
            train_idx = torch.tensor(idx[:int(0.6 * N)], dtype=torch.long)
            val_idx = torch.tensor(idx[int(0.6 * N):int(0.8 * N)], dtype=torch.long)
            test_idx = torch.tensor(idx[int(0.8 * N):], dtype=torch.long)
        else:
            train_idx = val_idx = test_idx = None  # will be created per seed

        seed_results = []


        for seed in tqdm(SEEDS, desc=f"Seeds k={k}"):
            # set seeds
            set_all_seeds(seed)

            # create dataset splits if not reusing
            if not REUSE_SPLITS_PER_K:
                idx = np.arange(N)
                np.random.shuffle(idx)
                train_idx = torch.tensor(idx[:int(0.6 * N)], dtype=torch.long)
                val_idx = torch.tensor(idx[int(0.6 * N):int(0.8 * N)], dtype=torch.long)
                test_idx = torch.tensor(idx[int(0.8 * N):], dtype=torch.long)

            # instantiate encoder
            enc = NodeEncoder(
                encoder="graphsage",
                in_dim=in_dim,
                hidden_dim=HYPERPARAMS["hidden_dim"],
                out_dim=HYPERPARAMS["hidden_dim"],
                num_layers=3,
                dropout=HYPERPARAMS["dropout"],
                use_input_mlp=True,
                activation=HYPERPARAMS["activation"],
                norm="batch",
                residual=True,
                jk_mode=HYPERPARAMS["jk_mode"],
                gat_heads=4,
                gat_concat=True,
                use_proj=True
            )

            # Reset GPU memory stats if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # TIME + RUN training
            t_start = time.perf_counter()
            enc, head, z_graphsage, losses, val_f1s, val_accs = train_node_classification(
                data, enc, train_idx, val_idx, test_idx,
                labels=labels, epochs=EPOCHS,
                lr=HYPERPARAMS["lr"],
                weight_decay=WEIGHT_DECAY,
                seed=seed,
                save_dir=save_dir,
                texts=texts,
                id2group=id2group,
                target_col=target_col,
                hyperparams=HYPERPARAMS
            )
            t_end = time.perf_counter()
            runtime_sec = t_end - t_start

            # GPU memory peak
            if torch.cuda.is_available():
                peak_bytes = torch.cuda.max_memory_allocated()
                peak_mb = bytes_to_mb(peak_bytes)
            else:
                peak_mb = None



            # best metrics
            best_f1 = float(np.max(val_f1s)) if len(val_f1s) else None
            best_epoch = int(np.argmax(val_f1s)) if len(val_f1s) else None

            # Save seed-level JSON (compact)
            seed_data = {
                "seed": int(seed),
                "k": int(k),
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "final_loss": float(losses[-1]) if len(losses) else None,
                "val_f1s": [float(x) for x in val_f1s],
                "val_accs": [float(x) for x in val_accs],
                "losses": [float(x) for x in losses],
                "train_idx_len": int(train_idx.shape[0]),
                "val_idx_len": int(val_idx.shape[0]),
                "test_idx_len": int(test_idx.shape[0]),
                "hyperparameters": HYPERPARAMS.copy(),
                "target_col": target_col,
                "timestamp": datetime.now().isoformat(),
                "runtime_sec": float(runtime_sec),
                "peak_gpu_mem_mb": peak_mb,
                "embedding_time_sec": float(embed_time),
                "embedding_flops_estimate": embed_flops
            }

            if save_dir:
                seed_file = os.path.join(save_dir, f"k{k}_seed_{seed}.json")
                with open(seed_file, "w") as f:
                    json.dump(seed_data, f, indent=2)
            print(f"Seed {seed} done — runtime: {runtime_sec:.2f}s, peak GPU MB: {peak_mb}, est total FLOPs: {estimated_total_flops:,d}")
            seed_results.append(seed_data)

        # Summarize per-k
        df_res = pd.DataFrame(seed_results)
        mean_f1 = df_res["best_f1"].mean()
        std_f1 = df_res["best_f1"].std()
        mean_runtime = df_res["runtime_sec"].mean()
        mean_gpu = df_res["peak_gpu_mem_mb"].mean() if df_res["peak_gpu_mem_mb"].notna().any() else None

        summary = {
            "k": int(k),
            "n_nodes": int(N),
            "n_edges": int(E),
            "avg_deg": float(avg_deg),
            "mean_best_f1": float(mean_f1),
            "std_best_f1": float(std_f1),
            "mean_runtime_sec": float(mean_runtime),
            "mean_peak_gpu_mem_mb": float(mean_gpu) if mean_gpu is not None else None,
        }

        results_by_k[k] = {"summary": summary, "seed_results": seed_results}

        # Print compact summary
        print("\n=== Summary for k={} ===".format(k))
        print(f"Avg Best F1: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"Avg runtime/seed: {mean_runtime:.1f}s")
        if mean_gpu is not None:
            print(f"Avg peak GPU mem/seed: {mean_gpu:.1f} MB")
        print(f"Estimated total FLOPs (all seeds): {summary['approx_total_flops_all_seeds']:,d}")

        # Save summary per k
        if save_dir:
            with open(os.path.join(save_dir, f"summary_k{k}.json"), "w") as f:
                json.dump(summary, f, indent=2)

    return results_by_k

def main():
    df_GERMAN_1 = pd.read_excel("Test Datasets/DATA_GERMAN_1st_Annotator.xlsx")
    df_GERMAN_1.drop(columns=[f"Unnamed: {i}" for i in range(2,33)], inplace=True)
    df_GERMAN_1.columns = df_GERMAN_1.iloc[3]
    df_GERMAN_1 = df_GERMAN_1.iloc[4:]
    df_GERMAN_1 = df_GERMAN_1.reset_index(drop=True)

    df_GERMAN_2 = pd.read_excel("Test Datasets/DATA_GERMAN_2nd_Annotator.xlsx")
    df_GERMAN_2.drop(columns=[f"Unnamed: {i}" for i in range(2,33)], inplace=True)
    df_GERMAN_2.columns = df_GERMAN_2.iloc[3]
    df_GERMAN_2 = df_GERMAN_2.iloc[4:]
    df_GERMAN_2 = df_GERMAN_2.reset_index(drop=True)


    ##### GERMAN 1
    print("\n\n\n\n")

    print("\n\n\n######### GERMAN 1 #########\n")
    results = run_graphsage_experiment(
            df=df_GERMAN_1,
            target_col=TARGET,
            embed_texts=embed_texts,
            build_knn_graph=build_knn_graph,
            add_attribute_co_membership_edges=add_attribute_co_membership_edges,
            NodeEncoder=NodeEncoder,
            train_node_classification=train_node_classification,
            HYPERPARAMS=HYPERPARAMS,
            EPOCHS=EPOCHS,
            WEIGHT_DECAY=WEIGHT_DECAY,
            SEEDS=SEEDS,
            folder_name="german_1_tensor_profiled",
    )

    ####### GERMAN 2
    print("\n\n\n\n")
    print("######### GERMAN 2 #########")


    results = run_graphsage_experiment(
        df=df_GERMAN_2,
        target_col=TARGET,
        embed_texts=embed_texts,
        build_knn_graph=build_knn_graph,
        add_attribute_co_membership_edges=add_attribute_co_membership_edges,
        NodeEncoder=NodeEncoder,
        train_node_classification=train_node_classification,
        HYPERPARAMS=HYPERPARAMS,
        EPOCHS=EPOCHS,
        WEIGHT_DECAY=WEIGHT_DECAY,
        SEEDS=SEEDS,
        folder_name="german_2",
    )

    #### FRENCH

    df_FRENCH_1 = pd.read_excel("Test Datasets/DATA_FRENCH_1st_Annotator.xlsx")
    df_FRENCH_1.drop(columns=[f"Unnamed: {i}" for i in range(2,33)], inplace=True)
    df_FRENCH_1.drop(columns=["Unnamed: 0" ], inplace=True)
    df_FRENCH_1.columns = df_FRENCH_1.iloc[3]
    df_FRENCH_1 = df_FRENCH_1.iloc[4:]
    df_FRENCH_1 = df_FRENCH_1.reset_index(drop=True)

    df_FRENCH_2 = pd.read_excel("Test Datasets/DATA_FRENCH_2ND_Annotator.xlsx")
    df_FRENCH_2.drop(columns=[f"Unnamed: {i}" for i in range(2,33)], inplace=True)
    df_FRENCH_2.drop(columns=["Unnamed: 0" ], inplace=True)
    df_FRENCH_2.columns = df_FRENCH_2.iloc[3]
    df_FRENCH_2 = df_FRENCH_2.iloc[4:]
    df_FRENCH_2 = df_FRENCH_2.reset_index(drop=True)

    ##### FRENCH 1
    print("\n\n\n\n")
    print("######### FRENCH 1 #########")
    results = run_graphsage_experiment(
        df=df_FRENCH_1,
        target_col=TARGET,
        embed_texts=embed_texts,
        build_knn_graph=build_knn_graph,
        add_attribute_co_membership_edges=add_attribute_co_membership_edges,
        NodeEncoder=NodeEncoder,
        train_node_classification=train_node_classification,
        HYPERPARAMS=HYPERPARAMS,
        EPOCHS=EPOCHS,
        WEIGHT_DECAY=WEIGHT_DECAY,
        SEEDS=SEEDS,
        folder_name="french_1",
    )

    #@#### FRENCH 2
    print("\n\n\n\n")
    print("######### FRENCH 2 #########")
    results = run_graphsage_experiment(
        df=df_FRENCH_2,
        target_col=TARGET,
        embed_texts=embed_texts,
        build_knn_graph=build_knn_graph,
        add_attribute_co_membership_edges=add_attribute_co_membership_edges,
        NodeEncoder=NodeEncoder,
        train_node_classification=train_node_classification,
        HYPERPARAMS=HYPERPARAMS,
        EPOCHS=EPOCHS,
        WEIGHT_DECAY=WEIGHT_DECAY,
        SEEDS=SEEDS,
        folder_name="french_2",
    )


    ### SLOVENIAN

    df_SLOVENIAN_1 = pd.read_excel("Test Datasets/DATA_SLOVENE_1st_Annotator.xlsx")
    df_SLOVENIAN_1.drop(columns=[f"Unnamed: {i}" for i in range(5,33)], inplace=True)
    # df_SLOVENIAN_1.drop(columns=["Unnamed: 0" ], inplace=True)
    df_SLOVENIAN_1.columns = df_SLOVENIAN_1.iloc[3]
    df_SLOVENIAN_1 = df_SLOVENIAN_1.iloc[4:]
    df_SLOVENIAN_1.drop(columns=['Tweet, Created at', 'Tweet URL', 'User, screen name', 'User, name'], inplace=True)
    df_SLOVENIAN_1 = df_SLOVENIAN_1.reset_index(drop=True)
    df_SLOVENIAN_1.rename(columns={'Tweet, Text': 'Text' }, inplace=True)
    # print(df_SLOVENIAN_1.columns)

    df_SLOVENIAN_2 = pd.read_excel("Test Datasets/DATA_SLOVENE_2nd_Annotator.xlsx")
    df_SLOVENIAN_2.drop(columns=[f"Unnamed: {i}" for i in range(5,33)], inplace=True)
    # df_SLOVENIAN_2.drop(columns=["Unnamed: 0" ], inplace=True)
    df_SLOVENIAN_2.columns = df_SLOVENIAN_2.iloc[3]
    df_SLOVENIAN_2 = df_SLOVENIAN_2.iloc[4:]
    df_SLOVENIAN_2.drop(columns=['Tweet, Created at', 'Tweet URL', 'User, screen name', 'User, name'], inplace=True)
    df_SLOVENIAN_2 = df_SLOVENIAN_2.reset_index(drop=True)
    df_SLOVENIAN_2.rename(columns={'Tweet, Text': 'Text' }, inplace=True)
    df_SLOVENIAN_2.dropna(subset=['Text'], inplace=True)


    ##### SLOVENIAN 1
    print("\n\n\n\n")
    print("######### SLOVENIAN 1 #########")
    results = run_graphsage_experiment(
        df=df_SLOVENIAN_1,
        target_col=TARGET,
        embed_texts=embed_texts,
        build_knn_graph=build_knn_graph,
        add_attribute_co_membership_edges=add_attribute_co_membership_edges,
        NodeEncoder=NodeEncoder,
        train_node_classification=train_node_classification,
        HYPERPARAMS=HYPERPARAMS,
        EPOCHS=EPOCHS,
        WEIGHT_DECAY=WEIGHT_DECAY,
        SEEDS=SEEDS,
        folder_name="slovenian_1"
    )

    ##### SLOVENIAN 2
    print("\n\n\n\n")
    print("######### SLOVENIAN 2 #########")
    df_SLOVENIAN_2.dropna(subset=['Text'], inplace=True)

    results = run_graphsage_experiment(
        df=df_SLOVENIAN_2,
        target_col=TARGET,
        embed_texts=embed_texts,
        build_knn_graph=build_knn_graph,
        add_attribute_co_membership_edges=add_attribute_co_membership_edges,
        NodeEncoder=NodeEncoder,
        train_node_classification=train_node_classification,
        HYPERPARAMS=HYPERPARAMS,
        EPOCHS=EPOCHS,
        WEIGHT_DECAY=WEIGHT_DECAY,
        SEEDS=SEEDS,
        folder_name="slovenian_2",
    )


if __name__ == "__main__":
    main()