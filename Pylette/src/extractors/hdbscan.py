import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from Pylette.src.color import Color
from Pylette.src.extractors.protocol import NP_T, ColorExtractorBase


class HDBSCANExtractor(ColorExtractorBase):
    @override
    def extract(self, arr: NDArray[NP_T], height: int, width: int, palette_size: int, **kwargs) -> list[Color]:
        """
        Extracts a color palette using HDBSCAN.

        Parameters:
            arr (NDArray[float]): The input array.
            height (int): The height of the image.
            width (int): The width of the image.
            palette_size (int): The number of colors to extract from the image.
            **kwargs: Additional keyword arguments to pass to HDBSCAN model.

        Returns:
            list[Color]: A palette of colors sorted by frequency.
        """

        from sklearn.cluster import HDBSCAN

        arr = np.squeeze(arr)
        # Set default parameters, allow overrides via kwargs
        hdbscan_params = {"min_cluster_size": int(len(arr) / 10), "allow_single_cluster": True, "copy": True}
        hdbscan_params.update(kwargs)
        model = HDBSCAN(**hdbscan_params)
        labels = model.fit_predict(arr)
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_freq = counts / counts.sum()

        colors = []
        for label, freq in zip(unique_labels, label_freq):
            if label == -1:
                continue  # Skip noise
            cluster_points = arr[labels == label]
            centroid = np.mean(cluster_points, axis=0).astype(int)
            colors.append(Color(centroid, freq))

        colors.sort(key=lambda c: c.freq, reverse=True)
        return colors[:palette_size]

def hdbscan_extraction(arr: NDArray[NP_T], height: int, width: int, palette_size: int, **kwargs) -> list[Color]:
    """
    Extracts a color palette using HDBSCAN.

    Parameters:
        arr (NDArray[float]): The input array.
        height (int): The height of the image.
        width (int): The width of the image.
        palette_size (int): The number of colors to extract from the image.
        **kwargs: Additional keyword arguments to pass to HDBSCAN model.

    Returns:
        list[Color]: A palette of colors sorted by frequency.
    """
    return HDBSCANExtractor().extract(arr=arr, height=height, width=width, palette_size=palette_size, **kwargs)