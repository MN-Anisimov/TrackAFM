import numpy as np
from skimage import morphology, filters
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def skeletonize_afm_mask(mask, min_branch_length=5):
    # Ensure binary input
    binary = mask > 0
    
    # Skeletonization
    skeleton = morphology.skeletonize(binary)
    
    # Remove small branches
    cleaned = morphology.remove_small_objects(skeleton, min_size=min_branch_length)
    
    return skeleton

def plot_skeleton_results(original, mask, skeleton):
    """Visualize skeletonization process with color rulers"""
    fig, (ax1, ax2, ax4) = plt.subplots(1, 3, figsize=(12, 10))
    
    # Original AFM image
    im1 = ax1.imshow(original, cmap='afmhot')
    ax1.set_title('Original AFM')
    add_color_ruler(im1, ax1, 'Height (nm)')
    
    # Binary mask
    im2 = ax2.imshow(mask, cmap='binary')
    ax2.set_title('Binary Mask')
    add_binary_ruler(im2, ax2)
    
    # Skeleton overlay
    overlay = original.copy()
    overlay[skeleton] = np.max(original)  # Highlight skeleton
    im4 = ax4.imshow(overlay, cmap='afmhot')
    ax4.set_title('Skeleton Overlay')
    add_color_ruler(im4, ax4, 'Height (nm)')
    
    # Mark skeleton points
    y, x = np.where(skeleton)
    ax4.scatter(x, y, color='cyan', s=1, alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# Helper functions for color rulers
def add_color_ruler(im, ax, label):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(label, rotation=270, labelpad=15)
    ax.axis('off')

def add_binary_ruler(im, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, ticks=[0, 1])
    cbar.set_ticklabels(['Background', 'Feature'])
    ax.axis('off')
