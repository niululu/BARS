# import json
# import os
# import matplotlib.pyplot as plt
# mpl.rcParams["font.family"] = "serif"
# mpl.rcParams["font.serif"] = ["DejaVu Serif"]
# mpl.rcParams["mathtext.fontset"] = "dejavuserif"
# # Update this path if needed
# JSON_PATH = "gas_results.json"
# OUT_DIR = "figures"
# OUT_PDF = os.path.join(OUT_DIR, "gas_cost_vs_n.pdf")
# OUT_PNG = os.path.join(OUT_DIR, "gas_cost_vs_n.png")
#
# def main():
#     if not os.path.exists(JSON_PATH):
#         raise FileNotFoundError(f"Cannot find {JSON_PATH}. Put it next to this script or update JSON_PATH.")
#
#     with open(JSON_PATH, "r") as f:
#         data = json.load(f)
#
#     Ns = [d["N"] for d in data]
#     avg_gas = [d["avgGas"] for d in data]
#
#     os.makedirs(OUT_DIR, exist_ok=True)
#
#     plt.figure()
#     plt.plot(Ns, avg_gas, marker="o")
#     plt.xlabel("Number of sequential enrollments (N)")
#     plt.ylabel("Average gas per enrollment (gas units)")
#     plt.xscale("log")  # makes 100..10000 easier to read; remove if you prefer linear
#     plt.tight_layout()
#
#     plt.savefig(OUT_PDF)
#     plt.savefig(OUT_PNG, dpi=300)
#     plt.close()
#
#     print(f"Saved: {OUT_PDF}")
#     print(f"Saved: {OUT_PNG}")
#
# if __name__ == "__main__":
#     main()
import json
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------
# Global font settings (IEEE-friendly)
# --------------------
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "dejavuserif"

mpl.rcParams["axes.labelsize"] = 14      # axis label font size
mpl.rcParams["xtick.labelsize"] = 12     # tick label size
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["legend.fontsize"] = 12
mpl.rcParams["axes.titlesize"] = 14

# --------------------
# Paths
# --------------------
JSON_PATH = "gas_results.json"
OUT_DIR = "figures"
OUT_PDF = os.path.join(OUT_DIR, "gas_cost_vs_n.pdf")
OUT_PNG = os.path.join(OUT_DIR, "gas_cost_vs_n.png")


def main():
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"Cannot find {JSON_PATH}. Put it next to this script or update JSON_PATH.")

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    Ns = [d["N"] for d in data]
    avg_gas = [d["avgGas"] for d in data]

    os.makedirs(OUT_DIR, exist_ok=True)

    # --------------------
    # Plot
    # --------------------
    plt.figure(figsize=(8.0, 5.5))

    plt.plot(
        Ns,
        avg_gas,
        marker="o",
        linewidth=2.0,
        markersize=6
    )

    plt.xlabel("Number of sequential enrollments (N)")
    plt.ylabel("Average gas per enrollment (gas units)")

    plt.xscale("log")  # log scale improves readability for 100–10000
    plt.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()

    plt.savefig(OUT_PDF)
    plt.savefig(OUT_PNG, dpi=300)
    plt.close()

    print(f"Saved: {OUT_PDF}")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
