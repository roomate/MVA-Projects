import numpy as np

# Compute HOG features of a image of shape 32x32 pixels

def hog_features(image, n_bins, pixel_cell, cell_block):
    # Compute gradients along each axis
    grad_1, grad_2 = np.gradient(image)
    # Compute the Euclidean distance between (grad_1, grad_2) and (0,0)
    magni = np.sqrt(grad_1 ** 2 + grad_2 ** 2)
    # Compute direction of the gradient 
    orient = (np.rad2deg(np.arctan2(grad_2, grad_1))) % 180

    # Compute the number of blocks in each dimension of the image
    num_cells_1 = image.shape[1] // pixel_cell[1]
    num_cells_2 = image.shape[0] // pixel_cell[0]
    num_blocks_1 = num_cells_1 - cell_block[1] + 1
    num_blocks_2 = num_cells_2 - cell_block[0] + 1
    # Total number of bins
    num_bins = n_bins * cell_block[0] * cell_block[1]
    
    features = np.zeros((num_blocks_2, num_blocks_1, num_bins))
    # Compute HOG and block normalization
    for i in range(num_blocks_2):
        for j in range(num_blocks_1):
            feat_bloc = []
            for m in range(cell_block[0]):
                for n in range(cell_block[1]):
                    # Extract the magnitude and orientation of the gradients for the current cell
                    cell_magni = magni[(i + m) * pixel_cell[0]: (i + m + 1) * pixel_cell[0],
                                       (j + n) * pixel_cell[1]: (j + n + 1) * pixel_cell[1]]
                    cell_orient = orient[(i + m) * pixel_cell[0]: (i + m + 1) * pixel_cell[0],
                                         (j + n) * pixel_cell[1]: (j + n + 1) * pixel_cell[1]]
                    # Compute the number of pixels in the current cell
                    num_pixel_cell = cell_magni.shape[0] * cell_magni.shape[1]
                    # Compute the histogram of oriented gradients for the current cell
                    hog_cell, _ = np.histogram(cell_orient, bins=n_bins, weights=cell_magni)
                    # Normalize the histogram of oriented gradients
                    feat_bloc.extend(hog_cell / num_pixel_cell)
            # Normalize block features (ensure to not divide by 0)
            if np.linalg.norm(feat_bloc) == 0:
                feat_bloc = feat_bloc
            else:
                feat_bloc = feat_bloc / np.linalg.norm(feat_bloc)
            # Assign the normalized block features to the features array
            features[i, j] = feat_bloc = feat_bloc
    # Flatten the features
    return features.ravel()
