import eval
import logging
import time
from custom_types import (
    TopicSegmentationAlgorithm,
    TopicSegmentationDatasets,
    TopicSegmentationConfig,
    TextTilingHyperparameters,
)


def main():
    # logging.root.setLevel(logging.INFO)
    eval.eval_topic_segmentation(
        TopicSegmentationDatasets.AMI,
        TopicSegmentationAlgorithm.SBERT,
        TopicSegmentationConfig(TextTilingHyperparameters()),
    )


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time)
