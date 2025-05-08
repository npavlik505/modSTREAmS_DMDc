# #
# # SCRIPT FOR ANIMATIONS WITH MESH
# #

# #!/usr/bin/env python3
# """
# animate_dmdc.py

# Animate the span_average dataset in dmdc_span_averages.h5
# (shaped [T=500, 1, nx=600, ny=208]) and either display interactively
# or save to disk as MP4/GIF.
# """

# import argparse
# import os

# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# Velocity_Component = 3 # U, V, W correspond to 0, 1, 2 in dmdc_span_averages.h5 and 1, 2, 3 in ./output/distribute_save/span_averages.h5

# def main():
#     p = argparse.ArgumentParser(
#         description="Animate span_average from a DMDc‐reconstructed HDF5 file"
#     )
#     p.add_argument(
#         "-f", "--file",
#         default="dmdc_span_averages.h5",
#         help="Path to the HDF5 file"
#     )
#     p.add_argument(
#         "-d", "--dataset",
#         default="span_average",
#         help="Name of the 4D dataset"
#     )
#     p.add_argument(
#         "--cmap",
#         default="viridis",
#         help="Matplotlib colormap"
#     )
#     p.add_argument(
#         "--fps",
#         type=int,
#         default=60,
#         help="Frames per second when saving"
#     )
#     p.add_argument(
#         "-s", "--save",
#         metavar="OUT.mp4",
#         help="Filename to save the animation (MP4/GIF). If omitted, show interactively."
#     )
#     args = p.parse_args()

#     if not os.path.isfile(args.file):
#         raise FileNotFoundError(f"File not found: {args.file}")

#     # load the mesh
#     mesh_path = os.path.join(os.path.dirname(args.file), "mesh.h5")
#     if not os.path.isfile(mesh_path):
#         raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
#     with h5py.File(mesh_path, "r") as mf:
#         # shape (1,600) → flatten to (600,)
#         x_grid = mf["x_grid"][:].reshape(-1)
#         # shape (1,208) → flatten to (208,)
#         y_grid = mf["y_grid"][:].reshape(-1)
#     # build a 2D mesh for plotting
#     X, Y = np.meshgrid(x_grid, y_grid)

#     # load the entire span-average dataset into memory
#     with h5py.File(args.file, "r") as f:
#         if args.dataset not in f:
#             raise KeyError(f"Dataset '{args.dataset}' not in {args.file}")
#         dset = f[args.dataset]
#         # dset.shape == (500,1,600,208)
#         raw = dset[:, Velocity_Component, :, :]           # -> (500,600,208)

#     # transpose each frame so that x runs horizontal
#     data = raw.transpose(0, 2, 1)       # -> (500,208,600)

#     T, ny, nx = data.shape
#     vmin, vmax = data.min(), data.max()

#     fig, ax = plt.subplots()
#     # use pcolormesh so that each cell is drawn at the correct (x,y) location
#     quad = ax.pcolormesh(
#         X, Y, data[0],
#         shading="auto",
#         cmap=args.cmap,
#         vmin=vmin, vmax=vmax
#     )
#     title = ax.set_title("t = 0")
#     cb = fig.colorbar(quad, ax=ax)
#     cb.set_label(args.dataset)
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")

#     def update(frame):
#         # update the QuadMesh's values
#         quad.set_array(data[frame].ravel())
#         title.set_text(f"t = {frame}")
#         return quad, title

#     ani = animation.FuncAnimation(
#         fig, update,
#         frames=T,
#         interval=1000/args.fps,
#         blit=False,
#         repeat=False,
#     )

#     if args.save:
#         ext = os.path.splitext(args.save)[1].lower()
#         if ext in (".mp4", ".mov"):
#             from matplotlib.animation import FFMpegWriter
#             writer = FFMpegWriter(
#                 fps=args.fps,
#                 metadata=dict(artist='DMDc'),
#                 bitrate=1800
#             )

#             # manually drive the save loop
#             with writer.saving(fig, args.save, dpi=fig.dpi):
#                 for i in range(T):
#                     # update the QuadMesh & title
#                     quad.set_array(data[i].ravel())
#                     title.set_text(f"t = {i}")

#                     # *** force a draw of the figure ***
#                     fig.canvas.draw()

#                     # now grab the updated frame
#                     writer.grab_frame()

#         elif ext == ".gif":
#             ani.save(args.save, writer='pillow', fps=args.fps)
#         else:
#             raise ValueError("Use .mp4 or .gif for --save")
#         print(f"Animation written to {args.save}")
#     else:
#         plt.show()




# if __name__ == "__main__":
#     main()



#
# SCRIPT FOR ANIMATION WITHOUT MESH
#


#!/usr/bin/env python3
"""
animate_dmdc.py

Animate the span_average dataset in dmdc_span_averages.h5
(shaped [T=500, 1, nx=600, ny=208]) and either display interactively
or save to disk as MP4/GIF.
"""

import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

Velocity_Component = 0 # U, V, W correspond to 0, 1, 2 in dmdc_span_averages.h5 and 1, 2, 3 in ./output/distribute_save/span_averages.h5

def main():
    p = argparse.ArgumentParser(
        description="Animate span_average from a DMDc‐reconstructed HDF5 file"
    )
    p.add_argument(
        "-f", "--file",
        default="output/distribute_save/span_averages.h5", # Either "dmdc_span_averages.h5" or "output/distribute_save/span_averages.h5"
        help="Path to the HDF5 file"
    )
    p.add_argument(
        "-d", "--dataset",
        default="span_average",
        help="Name of the 4D dataset"
    )
    p.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap"
    )
    p.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frames per second when saving"
    )
    p.add_argument(
        "-s", "--save",
        metavar="OUT.mp4",
        help="Filename to save the animation (MP4/GIF). If omitted, show interactively."
    )
    args = p.parse_args()

    if not os.path.isfile(args.file):
        raise FileNotFoundError(f"File not found: {args.file}")

    # --- no mesh reading any more! ---

    # load the entire span-average dataset into memory
    with h5py.File(args.file, "r") as f:
        if args.dataset not in f:
            raise KeyError(f"Dataset '{args.dataset}' not in {args.file}")
        dset = f[args.dataset]
        # dset.shape == (500,1,600,208)
        raw = dset[:, Velocity_Component, :, :]  # -> (500,600,208)

    # transpose each frame so that x runs horizontal
    data = raw.transpose(0, 2, 1)  # -> (500,208,600)

    T, ny, nx = data.shape
    vmin, vmax = data.min(), data.max()

    fig, ax = plt.subplots()
    # back to imshow
    im = ax.imshow(
        data[0],
        origin="lower",
        aspect="auto",
        cmap=args.cmap,
        vmin=vmin, vmax=vmax
    )
    title = ax.set_title("t = 0")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(args.dataset)
    ax.set_xlabel("x-index")
    ax.set_ylabel("y-index")

    def update(frame):
        im.set_data(data[frame])
        title.set_text(f"t = {frame}")
        return im, title

    ani = animation.FuncAnimation(
        fig, update,
        frames=T,
        interval=1000/args.fps,
        blit=False,
        repeat=False,
    )

    if args.save:
        ext = os.path.splitext(args.save)[1].lower()
        if ext in (".mp4", ".mov"):
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(
                fps=args.fps,
                metadata=dict(artist='DMDc'),
                bitrate=1800
            )
            # manually drive the save loop
            with writer.saving(fig, args.save, dpi=fig.dpi):
                for i in range(T):
                    im.set_data(data[i])
                    title.set_text(f"t = {i}")
                    fig.canvas.draw()
                    writer.grab_frame()

        elif ext == ".gif":
            ani.save(args.save, writer='pillow', fps=args.fps)
        else:
            raise ValueError("Use .mp4 or .gif for --save")
        print(f"Animation written to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()