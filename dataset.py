import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt

def generate_bars_and_stripes(num_samples):
    """
    Generate the Bars and Stripes dataset.

    Parameters:
        num_samples (int): Number of samples to generate.

    Returns:
        dataset (numpy.ndarray): Array of shape (num_samples, 4, 4) containing the patterns.
        labels (numpy.ndarray): Array of shape (num_samples,) containing the corresponding labels (0 for bars, 1 for stripes).
    """
    dataset = []
    labels = []
    for _ in range(num_samples):
        # Randomly choose if it's a horizontal bar (0) or a vertical stripe (1)
        pattern_type = np.random.randint(2)

        if pattern_type == 0:
            # Horizontal bar pattern
            pattern = np.zeros((4, 4))
            row_idx = np.random.randint(4)
            pattern[row_idx, :] = 1
        else:
            # Vertical stripe pattern
            pattern = np.zeros((4, 4))
            col_idx = np.random.randint(4)
            pattern[:, col_idx] = 1

        dataset.append(pattern)
        labels.append(pattern_type)

    return np.array(dataset), np.array(labels)

# Example usage:
num_samples = 28
bars_and_stripes_dataset, labels = generate_bars_and_stripes(num_samples)

# # Visualization
# plt.figure(figsize=(12, 3*num_samples))

# for i in range(num_samples):
#     plt.subplot(num_samples, 1, i+1)
#     plt.imshow(bars_and_stripes_dataset[i], cmap='gray', vmin=0, vmax=1)
#     plt.axis('off')
#     plt.title(f"Pattern {i+1}")

# plt.tight_layout()
# plt.show()