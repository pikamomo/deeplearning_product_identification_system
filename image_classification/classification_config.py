FASHION_LABELS_PATH = "../common/fashion-labels.csv"
IMG_PATH = "../common/dataset/"
IMG_HEIGHT = 64
IMG_WiDTH = 64

SEED = 42
TRAIN_RATIO = 0.75
TEST_RATIO = 1 - TRAIN_RATIO

LEARNING_RATE = 1e-3
EPOCHS = 10
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

PACKAGE_NAME = "image_classification"
DENOISER_MODEL_NAME = "classifier.pt"


classification_names = {
    0: 'cloth',
    1: 'shoe',
    2: 'bag',
    3: 'pants',
    4: 'watch'
}
