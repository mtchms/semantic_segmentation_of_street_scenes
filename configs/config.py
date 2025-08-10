import os

ROOT_DIR = os.getenv('MAPILLARY_ROOT', './data')

TRAIN_SPLIT = 'training'
VAL_SPLIT   = 'validation'

PATCH_SIZE = 512       
BATCH_SIZE = 4         

NUM_CLASSES = 124

NUM_EPOCHS = 20        
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

FLIP_PROB = 0.5
ROTATION_DEGREES = 15 
COLOR_JITTER = {
    'brightness': 0.2,
    'contrast':   0.2,
    'saturation': 0.2,
    'hue':        0.1
}

FINETUNE_START_EPOCH = 5
FT_LEARNING_RATE = 1e-4

OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

