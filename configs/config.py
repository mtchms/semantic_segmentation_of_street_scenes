import os

# Путь к распакованному Mapillary Vistas v2.0:
# Можно задать через переменную окружения MAPILLARY_ROOT, иначе по умолчанию './data'
ROOT_DIR = os.getenv('MAPILLARY_ROOT', './data')

# Сplits
TRAIN_SPLIT = 'training'
VAL_SPLIT   = 'validation'

# Патчи и batch
PATCH_SIZE = 512       # размер патча для обучения
BATCH_SIZE = 4         # на CPU обычно 1, на GPU можно увеличить до 2–4

# Число семантических классов Mapillary Vistas v2.0
NUM_CLASSES = 124

# Обучение
NUM_EPOCHS = 20        # можно увеличить при необходимости
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# Аугментации (MapillaryDatasetAug)
FLIP_PROB = 0.5
ROTATION_DEGREES = 15  # ±15°
# Параметры color jitter (яркость, контраст, насыщенность в долях)
COLOR_JITTER = {
    'brightness': 0.2,
    'contrast':   0.2,
    'saturation': 0.2,
    'hue':        0.1
}

# Fine-tune (когда будет pretrained backbone; оставлено для будущего)
FINETUNE_START_EPOCH = 5
FT_LEARNING_RATE = 1e-4

# Куда сохранять модели и логи
OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
