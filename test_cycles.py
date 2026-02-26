import numpy as np

# Simulate a sine wave of 3 full cycles
t = np.linspace(0, 3*2*np.pi, 300)
H = -np.sin(t) # Starts at 0, goes negative, crosses positive at pi
# Zero crossings at pi, 3pi, 5pi. Wait, sin crosses at 0, pi, 2pi...
# Let's just make it simple:
H = np.array([-1, 1, 2, 1, -1, 1, 2, 1, -1, 1, 2, 1, -1, 1, 2, 1])

# H crosses 0 to positive at index 0->1, 4->5, 8->9, 12->13
# 4 crossings = 3 cycles.

def cut(data):
    sign_changes = np.diff(np.signbit(data))
    crossing_indices = np.where(sign_changes)[0]
    pos_slope_indices = []
    for idx in crossing_indices:
        if idx + 1 < len(data):
            if data[idx] < data[idx+1]:
                pos_slope_indices.append(idx)
    start_idx = pos_slope_indices[0]
    end_idx = pos_slope_indices[-1]
    print(f"cut pos_slope_indices: {pos_slope_indices}")
    return data[start_idx : end_idx]

def avg(data):
    sign_changes = np.diff(np.signbit(data))
    crossing_indices = np.where(sign_changes)[0]
    pos_slope_indices = []
    for idx in crossing_indices:
        if idx + 1 < len(data):
            if data[idx] < data[idx+1]:
                pos_slope_indices.append(idx)
    print(f"avg pos_slope_indices: {pos_slope_indices}")
    num_cycles = len(pos_slope_indices) - 1
    # We also need to add the end of the array as the last boundary!
    print(f"num_cycles: {num_cycles}")

h_cut = cut(H)
avg(h_cut)
