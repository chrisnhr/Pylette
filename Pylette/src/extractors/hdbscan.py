import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from Pylette.src.color import Color
from Pylette.src.extractors.protocol import NP_T, ColorExtractorBase


class HDBSCANExtractor(ColorExtractorBase):
    @override
    def extract(self, arr: NDArray[NP_T], height: int, width: int, palette_size: int) -> list[Color]:
        """
        Extracts a color palette using HDBSCAN.

        Parameters:
            arr (NDArray[float]): The input array.
            height (int): The height of the image.
            width (int): The width of the image.
            palette_size (int): The number of colors to extract from the image.

        Returns:
            list[Color]: A palette of colors sorted by frequency.
        """

        from sklearn.cluster import HDBSCAN

        arr = np.squeeze(arr)
        clusterer = HDBSCAN(min_cluster_size=max(2, len(arr) // (palette_size * 10)), prediction_data=True)
        labels = clusterer.fit_predict(arr)
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_freq = counts / counts.sum()

        colors = []
        for label, freq in zip(unique_labels, label_freq):
            if label == -1:
                continue  # Skip noise
            cluster_points = arr[labels == label]
            centroid = np.mean(cluster_points, axis=0).astype(int)
            colors.append(Color(centroid, freq))

        colors.sort(key=lambda c: c.frequency, reverse=True)
        return colors[:palette_size]