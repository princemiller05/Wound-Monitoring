"""
Plot helpers for healing prediction output.
Uses the 'Agg' backend so it works in headless environments (servers, Colab, etc).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # must be set BEFORE importing pyplot
import matplotlib.pyplot as plt


def plot_mask_progression(masks_dict: dict, save_path: str, case_id: str = ""):
    """Side-by-side mask images showing wound shrinking (or not) over time."""
    days = sorted(masks_dict.keys())
    fig, axes = plt.subplots(1, len(days), figsize=(3 * len(days), 3))

    # If only one day, axes is a single Axes, not a list — wrap it
    if len(days) == 1:
        axes = [axes]

    for ax, day in zip(axes, days):
        ax.imshow(masks_dict[day], cmap="gray")
        ax.set_title(f"Day {day}")
        ax.axis("off")

    if case_id:
        fig.suptitle(f"{case_id} - Mask Progression")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close(fig)   # close so memory doesn't pile up over many runs


def plot_area_trend(longitudinal_df, save_path: str, case_id: str = ""):
    """Line plot of area vs day. A good wound trends down."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(longitudinal_df["day"], longitudinal_df["area_pixels"], marker="o")
    ax.set_xlabel("Day")
    ax.set_ylabel("Area (pixels)")
    ax.set_title(f"Wound Area Progression{f' - {case_id}' if case_id else ''}")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_tissue_trend(longitudinal_df, save_path: str, case_id: str = ""):
    """Three-line plot showing how each tissue type evolves over days."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(longitudinal_df["day"], longitudinal_df["granulation_pct"],
            marker="o", label="Granulation")
    ax.plot(longitudinal_df["day"], longitudinal_df["slough_pct"],
            marker="o", label="Slough")
    ax.plot(longitudinal_df["day"], longitudinal_df["necrosis_pct"],
            marker="o", label="Necrosis")
    ax.set_xlabel("Day")
    ax.set_ylabel("Percent")
    ax.set_title(f"Tissue Composition{f' - {case_id}' if case_id else ''}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
