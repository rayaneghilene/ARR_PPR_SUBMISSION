from torch_geometric.data import Data
from sklearn.metrics import f1_score
from Semi_supervised_approach.models.helper_methods import compute_auc
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json, os
import random
import torch.nn.functional as F 
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility. In the main we specify the seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_node_classification(
    data,
    encoder,
    train_idx,
    val_idx,
    test_idx,
    labels,
    epochs=50,
    lr=1e-3,
    weight_decay=1e-5,
    grad_clip: float = 2.0,
    seed: int = 42,
    verbose: bool = True,
    save_dir=None, 
    texts=None,  
    id2group=None, 
    target_col=None, 
    hyperparams=None,
):
    """
    Training loop for node classification using an encoder + linear head.

    Args:
        data: PyG Data object with x, edge_index
        encoder: NodeEncoder instance
        train_idx, val_idx, test_idx: torch.LongTensor node indices
        labels: torch.LongTensor of node labels
        grad_clip: max norm for gradient clipping
        seed: random seed for reproducibility
        verbose: print training progress
    """
    # Reproducibility
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    data = data.to(device)
    labels = labels.to(device)

    num_classes = int(labels.max().item()) + 1
    # head = nn.Linear(encoder.proj.out_features, num_classes).to(device)
    head = nn.Linear(encoder.out_dim, num_classes).to(device)

    # head = nn.Linear(encoder.proj.out_dim, num_classes).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0.0
    best_state = None
    losses, val_accs, val_f1s = [], [], []

    for ep in range(1, epochs + 1):
        encoder.train()
        head.train()
        optimizer.zero_grad()

        z = encoder(data.x, data.edge_index)  # [N, out_dim]
        logits = head(z)  # [N, num_classes]

        loss = loss_fn(logits[train_idx], labels[train_idx])
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), grad_clip)
        optimizer.step()

        # Evaluation
        encoder.eval()
        head.eval()
        with torch.no_grad():
            z = encoder(data.x, data.edge_index)
            logits = head(z)

            y_val_true = labels[val_idx].cpu().numpy()
            y_val_pred = logits[val_idx].argmax(dim=1).cpu().numpy()
            val_acc = (y_val_pred == y_val_true).mean()
            val_f1 = f1_score(y_val_true, y_val_pred, average="macro")

            if val_acc > best_val:
                best_val = val_acc
                best_state = {
                    "encoder": {k: v.cpu() for k, v in encoder.state_dict().items()},
                    "head": {k: v.cpu() for k, v in head.state_dict().items()},
                }

        if verbose and ep % 10 == 0:
            print(f"Epoch {ep:03d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            losses.append(loss.item())
            val_accs.append(val_acc)
            val_f1s.append(val_f1)

    # Load best weights
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        head.load_state_dict(best_state["head"])

    # Final test evaluation
    encoder.eval()
    head.eval()
    with torch.no_grad():
        z = encoder(data.x, data.edge_index)
        logits = head(z)
        y_test_true = labels[test_idx].cpu().numpy()
        y_test_pred = logits[test_idx].argmax(dim=1).cpu().numpy()
        test_acc = (y_test_pred == y_test_true).mean()
        test_f1 = f1_score(y_test_true, y_test_pred, average="macro")
        # Generate classification report for test set
        test_report = classification_report(y_test_true, y_test_pred, target_names=[f"Class {i}" for i in range(num_classes)])
        print("\nClassification Report (Test Set):")
        print(test_report)
        if save_dir is not None and texts is not None and id2group is not None:
            probs = F.softmax(logits, dim=1).cpu().numpy()
            predictions = logits.argmax(dim=1).cpu().numpy()
            
            # Prepare detailed instance data for each split
            def prepare_split_data(indices, split_name):
                split_data = []
                for idx in indices.cpu().numpy():
                    instance = {
                        "index": int(idx),
                        "text": texts[idx],
                        "true_label": id2group[labels[idx].item()],
                        "true_label_id": int(labels[idx].item()),
                        "predicted_label": id2group[predictions[idx]],
                        "predicted_label_id": int(predictions[idx]),
                        "correct": bool(predictions[idx] == labels[idx].item()),
                        "confidence": float(probs[idx, predictions[idx]]),
                        "split": split_name
                    }
                    # Add probability for each class
                    for class_id, class_name in id2group.items():
                        instance[f"prob_{class_name}"] = float(probs[idx, class_id])
                    split_data.append(instance)
                return split_data
            
            # Collect all instances
            all_instances = []
            all_instances.extend(prepare_split_data(train_idx, "train"))
            all_instances.extend(prepare_split_data(val_idx, "val"))
            all_instances.extend(prepare_split_data(test_idx, "test"))
            
            # Save as JSON
            instances_file = os.path.join(save_dir, f"seed_{seed}_instances.json")
            with open(instances_file, 'w', encoding='utf-8') as f:
                json.dump(all_instances, f, indent=2, ensure_ascii=False)
            
            # Save as CSV (for easier viewing :) )
            instances_df = pd.DataFrame(all_instances)
            instances_csv = os.path.join(save_dir, f"seed_{seed}_instances.csv")
            instances_df.to_csv(instances_csv, index=False, encoding='utf-8')
            
            # Save misclassifications separately
            misclassified = [inst for inst in all_instances if not inst['correct']]
            if misclassified:
                misclass_file = os.path.join(save_dir, f"seed_{seed}_misclassified.json")
                with open(misclass_file, 'w', encoding='utf-8') as f:
                    json.dump(misclassified, f, indent=2, ensure_ascii=False)
                
                misclass_df = pd.DataFrame(misclassified)
                misclass_csv = os.path.join(save_dir, f"seed_{seed}_misclassified.csv")
                misclass_df.to_csv(misclass_csv, index=False, encoding='utf-8')
            
            # Save metrics and training info
            seed_data = {
                "seed": seed,
                "best_val_acc": float(best_val),
                "test_acc": float(test_acc),
                "test_f1": float(test_f1),
                "val_f1s": [float(x) for x in val_f1s],
                "val_accs": [float(x) for x in val_accs],
                "losses": [float(x) for x in losses],
                "train_idx": train_idx.cpu().tolist(),
                "val_idx": val_idx.cpu().tolist(),
                "test_idx": test_idx.cpu().tolist(),
                "hyperparameters": hyperparams if hyperparams is not None else {},
                "target_col": target_col,
                "num_classes": num_classes,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                # Add accuracy per split
                "train_accuracy": float((predictions[train_idx.cpu().numpy()] == labels[train_idx].cpu().numpy()).mean()),
                "val_accuracy": float((predictions[val_idx.cpu().numpy()] == labels[val_idx].cpu().numpy()).mean()),
                "test_accuracy": float((predictions[test_idx.cpu().numpy()] == labels[test_idx].cpu().numpy()).mean()),
            }
            
            metrics_file = os.path.join(save_dir, f"seed_{seed}_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(seed_data, f, indent=2)
            
            print(f"\n  → Saved instances to {instances_csv}")
            print(f"  → Saved {len(misclassified)} misclassifications to {misclass_csv}")
            print(f"  → Saved metrics to {metrics_file}")

    if verbose:
        print(f"Best Val Acc: {best_val:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

    return encoder, head, z.cpu().numpy(), losses, val_f1s, val_accs



################## link prediction training loop #####################

def train_link_prediction(data: Data, encoder: nn.Module, pos_edges: torch.Tensor, neg_edges: torch.Tensor, decoder: nn.Module, epochs=50, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device); decoder.to(device)
    data = data.to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        encoder.train(); decoder.train()
        opt.zero_grad()
        z = encoder(data.x, data.edge_index)
        pos = pos_edges.to(device)
        neg = neg_edges.to(device)
        scores = decoder(z, pos, neg)
        pos_logits = scores['pos']
        neg_logits = scores['neg']
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        logits = torch.cat([pos_logits, neg_logits])
        loss = bce(logits, labels)
        loss.backward()
        opt.step()
        if ep % 10 == 0:
            with torch.no_grad():
                pred = torch.sigmoid(logits)
                auc = compute_auc(labels.cpu().numpy(), pred.cpu().numpy())
            print(f"Epoch {ep} loss={loss.item():.4f} AUC≈{auc:.4f}")
    return encoder, decoder
