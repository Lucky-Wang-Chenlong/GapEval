#!/usr/bin/env python3
"""
You need to generate/understand the dataset from your own umm. and evaluate by the SOTA model
Evaluate fixed dual-lambda gaps from a JSON file.

Input JSON format:
{
  "model_name": [
    [p11, p10, p01, p00],
    [p11, p10, p01, p00],
    [p11, p10, p01, p00],
    [p11, p10, p01, p00]
  ]
}

Where:
- p11 means both tasks are done successfully
- p10 means und. task is done successfully, while gen. task fail
- p01 means gen. task is done successfully, while und. task fail
- p00 both tasks are done failed

The script writes a new JSON file next to the input file with the suffix "_gapres" in the same filefold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F



def load_input(path: Path) -> dict[str, list[list[float]]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Input JSON must be a non-empty object mapping model names to row lists.")
    return payload


def normalize_joint_rows(rows: list[list[float]], model_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    joint_rows = []
    co_success = []
    co_fail = []
    for idx, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"{model_name} row {idx} must be a list of four floats [p11, p10, p01, p00].")
        values = [float(x) for x in row]
        if any(x < 0.0 for x in values):
            raise ValueError(f"{model_name} row {idx} contains negative values: {values}")
        row_sum = sum(values)
        if abs(row_sum - 1.0) > 5e-3:
            raise ValueError(
                f"{model_name} row {idx} must sum to 1.0 in joint format; got sum={row_sum:.6f}, row={values}"
            )
        if row_sum <= 0.0:
            raise ValueError(f"{model_name} row {idx} has zero total mass: {values}")
        p11, p10, p01, p00 = [x / row_sum for x in values]
        text_success = p11 + p10
        text_fail = p01 + p00
        image_success = p11 + p01
        image_fail = p10 + p00
        joint_rows.append([text_fail, text_success, image_success, image_fail])
        co_success.append(p11)
        co_fail.append(p00)
    return (
        torch.tensor(joint_rows, dtype=torch.float32),
        torch.tensor(co_success, dtype=torch.float32),
        torch.tensor(co_fail, dtype=torch.float32),
    )


def fit_map_parameters(
    proportions: torch.Tensor,
    total_count: float,
    steps: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    torch.manual_seed(seed)
    eps = 1e-8
    proportions = proportions.to(device)
    counts = proportions * total_count
    text_fail = counts[:, 0]
    text_succ = counts[:, 1]
    image_succ = counts[:, 2]
    image_fail = counts[:, 3]

    n_text = text_fail + text_succ
    n_image = image_fail + image_succ
    p_text_obs = torch.clamp(text_succ / (n_text + eps), 1e-5, 1.0 - 1e-5)
    p_image_obs = torch.clamp(image_succ / (n_image + eps), 1e-5, 1.0 - 1e-5)
    theta_init = torch.stack([torch.logit(p_text_obs), torch.logit(p_image_obs)], dim=1)

    theta = torch.nn.Parameter(theta_init.clone())
    beta = torch.nn.Parameter(torch.zeros(2, device=device))
    mu = torch.nn.Parameter(theta_init.mean(dim=0))
    L_raw = torch.nn.Parameter(torch.tensor([[0.1, 0.0], [0.05, 0.1]], dtype=torch.float32, device=device))

    optimizer = torch.optim.AdamW([theta, beta, mu, L_raw], lr=lr, weight_decay=weight_decay)
    best_state = None
    best_loss = float("inf")

    for _ in range(steps):
        optimizer.zero_grad()
        L = torch.tril(L_raw)
        diag_idx = torch.arange(2, device=device)
        L = L.clone()
        L[diag_idx, diag_idx] = F.softplus(L[diag_idx, diag_idx]) + 1e-4
        sigma = L @ L.T
        sigma_inv = torch.linalg.inv(sigma)

        p_text = torch.sigmoid(theta[:, 0] - beta[0])
        p_image = torch.sigmoid(theta[:, 1] - beta[1])

        nll = -(
            text_succ * torch.log(p_text + eps)
            + text_fail * torch.log(1.0 - p_text + eps)
            + image_succ * torch.log(p_image + eps)
            + image_fail * torch.log(1.0 - p_image + eps)
        ).sum()

        diff = theta - mu
        mahal = 0.5 * torch.einsum("ni,ij,nj->n", diff, sigma_inv, diff).sum()
        logdet_term = proportions.shape[0] * torch.log(torch.diagonal(L)).sum()
        anchor_reg = 1e-3 * (beta.pow(2).sum() + mu.pow(2).sum())
        loss = nll + mahal + logdet_term + anchor_reg
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = theta.detach().clone()

    if best_state is None:
        raise RuntimeError("MAP fitting did not produce a valid state.")

    delta_raw = best_state[:, 0] - best_state[:, 1]
    gap_abs = delta_raw.abs() / (1.0 + delta_raw.abs())
    return gap_abs

LAMBDA_FAIL = 2.7
LAMBDA_SUCC = 1.05
def evaluate_model(
    rows: list[list[float]],
    model_name: str,
    total_count: float,
    map_steps: int,
    map_lr: float,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    proportions, co_success, co_fail = normalize_joint_rows(rows, model_name)
    gap_abs = fit_map_parameters(
        proportions=proportions,
        total_count=total_count,
        steps=map_steps,
        lr=map_lr,
        weight_decay=1e-4,
        seed=seed,
        device=device,
    )
    gap_abs = torch.clamp(gap_abs.to(device), 1e-6, 1.0 - 1e-6)
    co_success = co_success.to(device)
    co_fail = co_fail.to(device)
    adjusted_gap = torch.sigmoid(torch.logit(gap_abs) + LAMBDA_FAIL * co_fail - LAMBDA_SUCC * co_success)
    gaps = [round(float(x.detach().cpu()), 6) for x in adjusted_gap]
    return {
        "gaps": gaps,
        "gap_mean": round(sum(gaps) / len(gaps), 6),
    }


def build_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_gapres.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed dual-lambda gaps from a joint-probability JSON file.")
    parser.add_argument("input_json", type=str, help="Path to the input JSON file.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use, e.g. cpu or cuda:0.")
    parser.add_argument("--map-steps", type=int, default=2000, help="AdamW steps for MAP fitting per model.")
    parser.add_argument("--map-lr", type=float, default=5e-2, help="Learning rate for MAP fitting.")
    parser.add_argument("--total-count", type=float, default=1000.0, help="Scaling factor from proportions to counts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)

    input_path = Path(args.input_json).resolve()
    payload = load_input(input_path)

    results = {}
    for model_name, rows in payload.items():
        results[model_name] = evaluate_model(
            rows=rows,
            model_name=model_name,
            total_count=args.total_count,
            map_steps=args.map_steps,
            map_lr=args.map_lr,
            seed=args.seed,
            device=device,
        )

    output_payload = {
        "results": results,
    }

    output_path = build_output_path(input_path)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
