import torch
from anomalib.data import MVTecLOCO
from anomalib.models import EfficientAd
from anomalib.engine import Engine

torch.set_float32_matmul_precision("high")

# =========================
# عدّل هالسطرين فقط كل مرة
# =========================
CKPT_PATH = r"D:\project computer vision\outputs\efficientad_roi_bg\EfficientAd\MVTecLOCO\splicing_connectors\v1\weights\lightning\model.ckpt"
DATA_ROOT = r"D:\project computer vision\data_roi_bg"# =========================
# ثوابت
# =========================
CATEGORY = "splicing_connectors"
IMAGENET_DIR = r"D:\project computer vision\datasets\imagenette2-320\train"
EVAL_DIR = r"D:\project computer vision\outputs\checkpoint_eval_single"

datamodule = MVTecLOCO(
    root=DATA_ROOT,
    category=CATEGORY,
    train_batch_size=1,
    eval_batch_size=4,
    num_workers=0,
)

model = EfficientAd(
    imagenet_dir=IMAGENET_DIR
)

engine = Engine(
    default_root_dir=EVAL_DIR,
    accelerator="gpu",
    devices=1,
)

engine.test(
    model=model,
    datamodule=datamodule,
    ckpt_path=CKPT_PATH,
)