import numpy as np

def get_labels_colormap():
    """Return a colormap for distinct label colors based on:
    Green-Armytage, P., 2010. A colour alphabet and the limits of colour coding. JAIC-Journal of the International Colour Association, 5.
    """
    colormap_22 = {
        0: np.array([0, 0, 0, 0]),  # alpha
        1: np.array([1, 1, 0, 1]),  # yellow
        2: np.array([0.5, 0, 0.5, 1]),  # purple
        3: np.array([1, 0.65, 0, 1]),  # orange
        4: np.array([0.68, 0.85, 0.9, 1]),  # light blue
        5: np.array([1, 0, 0, 1]),  # red
        6: np.array([1, 0.94, 0.8, 1]),  # buff
        7: np.array([0.5, 0.5, 0.5, 1]),  # grey
        8: np.array([0, 0.5, 0, 1]),  # green
        9: np.array([0.8, 0.6, 0.8, 1]),  # purplish pink
        10: np.array([0, 0, 1, 1]),  # blue
        11: np.array([1, 0.85, 0.7, 1]),  # yellowish pink
        12: np.array([0.54, 0.17, 0.89, 1]),  # violet
        13: np.array([1, 0.85, 0, 1]),  # orange yellow
        14: np.array([0.65, 0.09, 0.28, 1]),  # purplish red
        15: np.array([0.68, 0.8, 0.18, 1]),  # greenish yellow
        16: np.array([0.65, 0.16, 0.16, 1]),  # reddish brown
        17: np.array([0.5, 0.8, 0, 1]),  # yellow green
        18: np.array([0.8, 0.6, 0.2, 1]),  # yellowish brown
        19: np.array([1, 0.27, 0, 1]),  # reddish orange
        20: np.array([0.5, 0.5, 0.2, 1]),  # olive green
    }
    return colormap_22
