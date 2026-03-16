import os
import json
import random
from time import time, sleep
import warnings

import numpy as np
import torch
from torch.nn.functional import l1_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from egnn_predictor.models import EGNN_predictor

from utils.utils_edm import (
    remove_mean_with_mask,
    assert_correctly_masked,
    assert_mean_zero_with_mask,
    normalize,
    MyDataParallel
)

from data.aromatic_dataloader import create_data_loaders, AromaticDataset
from prediction_args import PredictionArgs


warnings.simplefilter(action="ignore", category=FutureWarning)


def check_mask_correct(variables, node_mask):
    """
    Check that tensors are properly masked.
    """
    for variable in variables:
        if variable.shape[-1] != 0:
            assert_correctly_masked(variable, node_mask)



def compute_clean_loss(
    model,
    x,
    h,
    node_mask,
    edge_mask,
    target,
    adj_full,
):
    # 1) Normalize coordinates and node features with EDM
    x_norm, h_norm, _ = normalize(
        x,
        {"categorical": h, "integer": torch.zeros(0, device=x.device)},
        node_mask,
    )

    # 2) Build [x_norm | h_norm] input
    xh = torch.cat([x_norm, h_norm["categorical"]], dim=-1)  # [bs, n_nodes, d+in_nf]

    bs, n_nodes, _ = x.shape
    edge_mask_flat = edge_mask.view(bs, n_nodes * n_nodes)   # [bs, n_nodes^2]

    # 3) Forward pass
    preds = model(xh, node_mask, edge_mask_flat , adj_full)  # [bs, num_targets]

    # 5) L1 loss in normalized target space
    loss = l1_loss(preds, target)
    error = (preds - target).abs().detach()  # per-sample, per-target
    return loss, error


def train_epoch_clean(
    epoch,
    cond_predictor,
    dataloader,
    optimizer,
    args,
    writer,
):
    cond_predictor.train()
    start_time = time()
    loss_list = []
    rl_loss = []

    with tqdm(dataloader, unit="batch", desc=f"Train (clean) {epoch}") as tepoch:
        for i, (x, node_mask, edge_mask, node_features, y , adj_full) in enumerate(tepoch):
            x = x.to(args.device)
            y = y.to(args.device)
            node_mask = node_mask.to(args.device).unsqueeze(2)
            edge_mask = edge_mask.to(args.device)
            h = node_features.to(args.device)
            adj_full = adj_full.to(args.device)

            x = remove_mean_with_mask(x, node_mask)
            #TODO: correct asserts
            #check_mask_correct([x, h], node_mask)
            #assert_mean_zero_with_mask(x, node_mask)

            loss, _ = compute_clean_loss(
                cond_predictor,
                x,
                h,
                node_mask,
                edge_mask,
                y,
                adj_full,
            )

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            # Rescaled loss in original target units
            rl_loss.append(dataloader.dataset.rescale_loss(loss).item())

            tepoch.set_postfix(loss=np.mean(loss_list).item() )

    print(
        f"[{epoch}|train] loss: {np.mean(loss_list):.4f}+-{np.std(loss_list):.4f}, "
        f"L1 (rescaled): {np.mean(rl_loss):.4f}, "
        f" in {int(time()-start_time)} secs"
    )
    sleep(0.01)
    if writer is not None:
        writer.add_scalar("Train loss", np.mean(loss_list), epoch)
        writer.add_scalar("Train L1 (rescaled)", np.mean(rl_loss), epoch)


def val_epoch_clean(
    tag,
    epoch,
    cond_predictor,
    dataloader,
    args,
    writer,
):
    cond_predictor.eval()
    with torch.no_grad():
        start_time = time()
        loss_list = []
        rl_loss = []

        for i, (x, node_mask, edge_mask, node_features, y , adj_full) in enumerate(dataloader):
            x = x.to(args.device)
            y = y.to(args.device)
            node_mask = node_mask.to(args.device).unsqueeze(2)
            edge_mask = edge_mask.to(args.device)
            h = node_features.to(args.device)
            adj_full = adj_full.to(args.device)

            x = remove_mean_with_mask(x, node_mask)
            #check_mask_correct([x, h], node_mask)
            #assert_mean_zero_with_mask(x, node_mask)

            loss, _  = compute_clean_loss(
                cond_predictor,
                x,
                h,
                node_mask,
                edge_mask,
                y,
                adj_full,
            )

            loss_list.append(loss.item())
            rl_loss.append(dataloader.dataset.rescale_loss(loss).item())
            

        print(
            f"[{epoch}|{tag}] loss: {np.mean(loss_list):.4f}+-{np.std(loss_list):.4f}, "
            f"L1 (rescaled): {np.mean(rl_loss):.4f}, "
            f" in {int(time() - start_time)} secs"
        )
        sleep(0.01)
        if writer is not None:
            writer.add_scalar(f"{tag} loss", np.mean(loss_list), epoch)
            writer.add_scalar(f"{tag} L1 (rescaled)", np.mean(rl_loss), epoch)

    return np.mean(loss_list)


def get_cond_predictor_model(args, dataset: AromaticDataset):
    cond_predictor = EGNN_predictor(
        in_nf=dataset.num_node_features,
        device=args.device,
        hidden_nf=args.nf,
        out_nf=dataset.num_targets,
        act_fn=torch.nn.SiLU(),
        n_layers=args.n_layers,
        recurrent=True,
        tanh=args.tanh,
        attention=args.attention,
        coords_range=args.coords_range,
    )
    if args.dp:  # and torch.cuda.device_count() > 1:
        cond_predictor = MyDataParallel(cond_predictor)
    if args.restore is not None:
        checkpoint = torch.load(
            "/home/maayanfarkash/proj/prediction_summary/hetro/checkpoint_best.pt",
            map_location=args.device
        )
        cond_predictor.load_state_dict(checkpoint["model_state_dict"])
    return cond_predictor



def main(pred_args, device):
    # ---------------------------
    # Data
    # ---------------------------
    train_loader, val_loader, test_loader = create_data_loaders(pred_args)

    print("train_loader.num_workers =", train_loader.num_workers)
    print("val_loader.num_workers =", val_loader.num_workers)
    print("test_loader.num_workers =", test_loader.num_workers)

    print("Checking first 20 samples...")
    for i in range(min(20, len(train_loader.dataset))):
        print(f"sample {i}")
        sample = train_loader.dataset[i]

    print("Checking first 5 batches...")
    for i, batch in enumerate(train_loader):
        print(f"batch {i}")
        if i >= 4:
            break


    # ---------------------------
    # Predictor model
    # ---------------------------
    cond_predictor = get_cond_predictor_model(pred_args, train_loader.dataset)
    cond_predictor = cond_predictor.to(device)

##Maayan change
    # if pred_args.dp and torch.cuda.device_count() > 1:
    #     cond_predictor = torch.nn.DataParallel(cond_predictor)

    # ---------------------------
    # Optimizer
    # ---------------------------
    optimizer = torch.optim.Adam(
        cond_predictor.parameters(),
        lr=pred_args.lr,
        amsgrad=True, 
        weight_decay=1e-12,
    )

    start_epoch = 0
    best_val_mae = 1e9
    best_epoch = 0

    last_checkpoint_path = os.path.join(pred_args.exp_dir, "checkpoint_last.pt")

    if os.path.isfile(last_checkpoint_path):
        print(f"Loading checkpoint from {last_checkpoint_path}")
        checkpoint = torch.load(last_checkpoint_path, map_location=device)

        cond_predictor.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_mae = checkpoint.get("best_val_mae", 1e9)
        best_epoch = checkpoint.get("best_epoch", 0)

        print(f"Resuming training from epoch {start_epoch}")

    # ---------------------------
    # Logging & experiment dir
    # ---------------------------
    if not os.path.isdir(pred_args.exp_dir):
        os.makedirs(pred_args.exp_dir, exist_ok=True)

    with open(os.path.join(pred_args.exp_dir, "args_clean.txt"), "w") as f:
        json.dump(pred_args.__dict__, f, indent=2, default=str)


    writer = None
    if getattr(pred_args, "log_tensorboard", False):
        writer = SummaryWriter(log_dir=os.path.join(pred_args.exp_dir, "tb_clean"))

    print("predictor training args:")
    print(pred_args)

    # ---------------------------
    # Training loop
    # ---------------------------

    print("Begin training")
    for epoch in range(start_epoch, pred_args.num_epochs):
        train_epoch_clean(
            epoch,
            cond_predictor,
            train_loader,
            optimizer,
            pred_args,
            writer,
        )
        val_mae = val_epoch_clean(
            "val",
            epoch,
            cond_predictor,
            val_loader,
            pred_args,
            writer,
        )


        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": cond_predictor.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_mae": best_val_mae,
                    "best_epoch": best_epoch,
                },
                os.path.join(pred_args.exp_dir, "checkpoint_best.pt"),
            )
            print(f"Saved new best model at epoch {epoch} with MAE={val_mae:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": cond_predictor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_mae": best_val_mae,
                "best_epoch": best_epoch,
            },
            os.path.join(pred_args.exp_dir, "checkpoint_last.pt"),
        )


    print(f"Best val MAE: {best_val_mae:.4f} at epoch {best_epoch}")

    # ---------------------------
    # Final test evaluation
    # ---------------------------
    print("Testing best model...")
    best_ckpt_path = os.path.join(pred_args.exp_dir, "checkpoint_best.pt")
    if not os.path.isfile(best_ckpt_path):
        raise FileNotFoundError(f"Missing best checkpoint: {best_ckpt_path}")
    # reload best weights
    best_checkpoint = torch.load(
        os.path.join(pred_args.exp_dir, "checkpoint_best.pt"),
        map_location=device,
    )
    cond_predictor.load_state_dict(best_checkpoint["model_state_dict"])
    cond_predictor = cond_predictor.to(device)


    test_mae = val_epoch_clean(
        "test",
        best_epoch,
        cond_predictor,
        test_loader,
        pred_args,
        writer,
    )
    print(f"Final test MAE (orig units): {test_mae:.4f}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use same argument structure as cond_prediction
    pred_args = PredictionArgs().parse_args()

    # You can override some defaults here if you want:
    # pred_args.num_epochs = 100
    # pred_args.lr = 1e-4
    # pred_args.exp_dir = "prediction_summary/cond_predictor_clean/..."

    pred_args.device = device

    pred_args.exp_dir = f"{pred_args.save_dir}/{pred_args.name}"

    if not os.path.isdir(pred_args.exp_dir):
        os.makedirs(pred_args.exp_dir)

    #with open(os.path.join(pred_args.exp_dir, "args_clean.txt"), "w") as f:
        #json.dump(pred_args.__dict__, f, indent=2)
    with open(os.path.join(pred_args.exp_dir, "args_clean.txt"), "w") as f:
        args_dict = dict(pred_args.__dict__)
        if "device" in args_dict:
            args_dict["device"] = str(args_dict["device"])
        json.dump(args_dict, f, indent=2)

    main(pred_args, device)
