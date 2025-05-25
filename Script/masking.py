import numpy as np
import codecs
from scipy import ndimage
from skimage import filters
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skeleton import skeletonize_afm_mask, plot_skeleton_results


def open_im(path, num):
	with codecs.open(path + "\\image_" + str(num) + ".txt", encoding='utf-8-sig') as f:
		field = np.loadtxt(f)

	return field


def mask_skelet(afm_data, laplacian_data, threshold):
    mask = laplacian_data > threshold
    skeleton = skeletonize_afm_mask(mask, min_branch_length=10)
    plot_skeleton_results(afm_data, mask, skeleton)
    np.savetxt('afm_skeleton.txt', skeleton, fmt='%d')


def add_color_ruler(im, ax, label):
    """Add standard color ruler"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(label, rotation=270, labelpad=15)
    ax.axis('off')


def add_laplacian_ruler(im, ax, label):
    """Special ruler for Laplacian data (symmetric around zero)"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, format='%.1e')  # Scientific notation
    cbar.set_label(label, rotation=270, labelpad=15)
    
    # Highlight zero point
    cbar.ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.axis('off')


def add_binary_ruler(im, ax):
    """Simplified ruler for binary masks"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, ticks=[0, 1])
    cbar.set_ticklabels(['Background', 'Features'])
    ax.axis('off')


if __name__ == "__main__":

    image = open_im("C:\\Users\\amih1\\Documents\\Work\\Gwy_proc", 3)
    blurred = filters.gaussian(image, sigma=3)
    laplacian = -ndimage.laplace(blurred)
    mask_skelet(image, laplacian, threshold=2e-10)