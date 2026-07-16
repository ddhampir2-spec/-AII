# ==========================================================
# dcsk_system.py
# Differential Chaos Shift Keying
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt

from chua_generator import ChuaGenerator
from chua_generator import remove_transient

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

chaos, cut = remove_transient(
    x,
    window=1000,
    tolerance=0.01,
    consecutive=10
)

print("Transient removed at sample =", cut)

# Normalize

chaos = chua.normalize(chaos)

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
# Plot Chaotic Signal
# ==========================================================

plt.figure(figsize=(12,4))

plt.plot(
    chaos[:1000]
)

plt.title("Chaotic Signal")

plt.xlabel("Sample")

plt.ylabel("Amplitude")

plt.grid(True)

# ==========================================================
# Plot DCSK Signal
# ==========================================================

plt.figure(figsize=(12,4))

plt.plot(tx)

plt.title("DCSK Signal")

plt.xlabel("Sample")

plt.ylabel("Amplitude")

plt.grid(True)

# ==========================================================
# Plot First Symbol
# ==========================================================

plt.figure(figsize=(12,4))

plt.plot(
    tx[:2*SPREADING_FACTOR],
    linewidth=2
)

plt.axvline(
    SPREADING_FACTOR,
    color='red',
    linestyle='--'
)

plt.title("First DCSK Symbol")

plt.xlabel("Sample")

plt.ylabel("Amplitude")

plt.grid(True)

# ==========================================================
# Plot Reference and Data
# ==========================================================

plt.figure(figsize=(12,4))

plt.subplot(211)

plt.plot(
    reference[:SPREADING_FACTOR]
)

plt.title("Reference")

plt.grid(True)

plt.subplot(212)

plt.plot(
    information[:SPREADING_FACTOR]
)

plt.title("Information")

plt.grid(True)

plt.tight_layout()

plt.show()