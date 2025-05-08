#!/usr/bin/env python3
"""
view_slice.py

Open dmdc_span_averages.h5, extract a single (time,component) slice
from the 4D dataset span_average (shape: [500,1,600,208]) and display it.
"""

import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="View a single 2D slice from dmdc_span_averages.h5"
    )
    parser.add_argument(
        "--file", "-f",
        default="dmdc_span_averages.h5",
        help="Path to the HDF5 file"
    )
    parser.add_argument(
        "--dataset", "-d",
        default="span_average",
        help="Name of the dataset in the HDF5 file"
    )
    parser.add_argument(
        "--time", "-t",
        type=int,
        default=0,
        help="Time index (0 ≤ t < 500)"
    )
    parser.add_argument(
        "--component", "-c",
        type=int,
        default=0,
        help="Component index (0 ≤ c < 1)"
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap"
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Don’t pop up the interactive window"
    )
    parser.add_argument(
        "--save", "-s",
        metavar="OUT.png",
        help="If given, save the figure to this file"
    )
    args = parser.parse_args()

    # sanity check
    if not os.path.isfile(args.file):
        raise FileNotFoundError(f"Could not find {args.file}")

    with h5py.File(args.file, "r") as f:
        if args.dataset not in f:
            raise KeyError(f"Dataset '{args.dataset}' not found in {args.file}")
        dset = f[args.dataset]
        # expected shape: (500, 1, 600, 208)
        data = dset[args.time, args.component, :, :].T  # → (208,600)

    # plot
    fig, ax = plt.subplots()
    im = ax.imshow(
        data,
        origin="lower",     # so [0,0] is bottom‐left
        aspect="auto",
        cmap=args.cmap
    )
    ax.set_title(f"{args.dataset}[t={args.time},c={args.component}]")
    fig.colorbar(im, ax=ax, label="span average")

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Slice saved to {args.save}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
