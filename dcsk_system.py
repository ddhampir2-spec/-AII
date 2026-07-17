# ==========================================================
# dcsk_system.py
# Differential Chaos Shift Keying
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt

from chua_generator import (
    ChuaGenerator,
)

# ==========================================================
# PARAMETERS
# ==========================================================

SPREADING_FACTOR = 100

NUMBER_OF_BITS = 20

SIGMA = 0

SEED = 1

# ==========================================================
# Generate Chaotic Signal
# ==========================================================

chua = ChuaGenerator()

x, y, z = chua.simulate(
    sigma=SIGMA,
    seed=SEED
)

# Remove transient

x, y, z, cut = chua.remove_transient(
        x,
        y,
        z,
        transient_time=100     # có thể thay đổi
    )

print("Transient removed at sample =", cut)

# Normalize

chaos = chua.normalize(x)

# ==========================================================
# Generate Random Bits
# ==========================================================

np.random.seed(SEED)

bits = np.random.randint(
    0,
    2,
    NUMBER_OF_BITS
)

print("--------------------------------")
print("Transmitted Bits")
print("--------------------------------")

print(bits)

# ==========================================================
# DCSK Encoder
# ==========================================================

def dcsk_encode(
        chaos,
        bits,
        B):

    tx = []

    reference = []

    information = []

    index = 0

    for bit in bits:

        ref = chaos[index:index+B]

        if len(ref) < B:

            raise ValueError(
                "Chaotic sequence is too short."
            )

        if bit == 1:

            data = ref.copy()

        else:

            data = -ref

        reference.extend(ref)

        information.extend(data)

        tx.extend(ref)

        tx.extend(data)

        index += B

    return (
        np.array(reference),
        np.array(information),
        np.array(tx)
    )

# ==========================================================
# Encoding
# ==========================================================

reference, information, tx = dcsk_encode(

    chaos,
    bits,
    SPREADING_FACTOR

)

print("--------------------------------")
print("Encoding Finished")
print("--------------------------------")

print("Spreading Factor =", SPREADING_FACTOR)

print("Total Samples =", len(tx))

# ==========================================================
# NUMBER OF SAMPLES TO DISPLAY
# ==========================================================

display_samples = NUMBER_OF_BITS * 2 * SPREADING_FACTOR

chaos_display = chaos[:display_samples]

# ==========================================================
# CREATE FIGURE
# ==========================================================

chaos_samples = NUMBER_OF_BITS * SPREADING_FACTOR
dcsk_samples  = NUMBER_OF_BITS * 2 * SPREADING_FACTOR

fig, ax = plt.subplots(
    2,
    2,
    figsize=(18,10)
)

# ==========================================================
# 1. ORIGINAL CHAOTIC SIGNAL
# ==========================================================

ax[0,0].plot(
    chaos[:chaos_samples],
    linewidth=1
)

for k in range(NUMBER_OF_BITS):

    center = k*SPREADING_FACTOR + SPREADING_FACTOR/2

    ax[0,0].text(
        center,
        1.08,
        str(bits[k]),
        color='red',
        ha='center',
        fontsize=11,
        fontweight='bold'
    )

    ax[0,0].axvline(
        k*SPREADING_FACTOR,
        color='gray',
        alpha=0.4
    )

ax[0,0].set_title("Original Chaotic Signal")
ax[0,0].set_xlabel("Sample")
ax[0,0].set_ylabel("Amplitude")
ax[0,0].grid(True)

# ==========================================================
# 2. DCSK SIGNAL
# ==========================================================

ax[0,1].plot(
    tx[:dcsk_samples],
    linewidth=1
)

for k in range(NUMBER_OF_BITS):

    center = k*2*SPREADING_FACTOR + SPREADING_FACTOR

    ax[0,1].text(
        center,
        1.08,
        str(bits[k]),
        color='red',
        ha='center',
        fontsize=11,
        fontweight='bold'
    )

    # bắt đầu symbol
    ax[0,1].axvline(
        k*2*SPREADING_FACTOR,
        color='gray',
        alpha=0.4
    )

    # ranh giới Reference/Data
    ax[0,1].axvline(
        k*2*SPREADING_FACTOR + SPREADING_FACTOR,
        color='gray',
        linestyle='--',
        alpha=0.5
    )

ax[0,1].set_title("DCSK Encoded Signal")
ax[0,1].set_xlabel("Sample")
ax[0,1].set_ylabel("Amplitude")
ax[0,1].grid(True)

# ==========================================================
# 3. FIRST SYMBOL
# ==========================================================

ax[1,0].plot(
    tx[:2*SPREADING_FACTOR],
    linewidth=2
)

ax[1,0].axvline(
    SPREADING_FACTOR,
    color='red',
    linestyle='--',
    linewidth=2,
    label='Reference / Data'
)

ax[1,0].text(
    SPREADING_FACTOR/2,
    1.05,
    "Reference",
    ha='center',
    color='blue',
    fontsize=11
)

ax[1,0].text(
    SPREADING_FACTOR*1.5,
    1.05,
    "Data",
    ha='center',
    color='green',
    fontsize=11
)

ax[1,0].set_title(
    f"First DCSK Symbol (Bit = {bits[0]})"
)

ax[1,0].set_xlabel("Sample")
ax[1,0].set_ylabel("Amplitude")
ax[1,0].grid(True)
ax[1,0].legend()

# ==========================================================
# 4. REFERENCE vs INFORMATION
# ==========================================================

ax[1,1].plot(
    reference[:SPREADING_FACTOR],
    linewidth=2,
    label="Reference"
)

ax[1,1].plot(
    information[:SPREADING_FACTOR],
    '--',
    linewidth=2,
    label="Information"
)

ax[1,1].set_title(
    f"Reference vs Information (Bit = {bits[0]})"
)

ax[1,1].set_xlabel("Sample")
ax[1,1].set_ylabel("Amplitude")
ax[1,1].grid(True)
ax[1,1].legend()

# ==========================================================
# SHOW
# ==========================================================

plt.tight_layout()
plt.show()
