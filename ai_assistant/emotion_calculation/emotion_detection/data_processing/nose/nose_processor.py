from emotion_detection.data_processing.feature_processor import FeatureProcessor
from emotion_detection.data_processing.nose.nose_processing import (NosePointsProcessing,
                                                                    EuclideanDistanceCalculator)


class NoseProcessor(FeatureProcessor):
    def __init__(self):
        distance_calculator = EuclideanDistanceCalculator()
        self.processor = NosePointsProcessing(distance_calculator)

    def process(self, points: dict):
        return self.processor.main(points)
