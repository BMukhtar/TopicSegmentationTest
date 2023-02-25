import eval
from custom_types import (
    TopicSegmentationAlgorithm,
    TopicSegmentationDatasets,
    TopicSegmentationConfig,
    TextTilingHyperparameters,
)


def main():
    eval.eval_topic_segmentation(
        TopicSegmentationDatasets.AMI,
        TopicSegmentationAlgorithm.BERT,
        TopicSegmentationConfig(TextTilingHyperparameters()),
    )


if __name__ == "__main__":
    main()
