import eval
from custom_types import (
    TopicSegmentationAlgorithm,
    TopicSegmentationDatasets,
    TopicSegmentationConfig,
)


def main():
    eval.eval_topic_segmentation(
        TopicSegmentationDatasets.AMI,
        TopicSegmentationAlgorithm.EVEN,
        TopicSegmentationConfig(),
    )


if __name__ == "__main__":
    main()
