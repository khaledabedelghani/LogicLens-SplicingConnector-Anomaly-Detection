# import torch
# from anomalib.data import MVTecLOCO
# from anomalib.models import EfficientAd
# from anomalib.engine import Engine

# torch.set_float32_matmul_precision("high")

# CKPT_PATH = r"D:\project computer vision\outputs\efficientad\EfficientAd\MVTecLOCO\splicing_connectors\v21\weights\lightning\model.ckpt"

# datamodule = MVTecLOCO(
#     root=r"D:\project computer vision\data",
#     category="splicing_connectors",
#     train_batch_size=1,
#     eval_batch_size=4,
#     num_workers=0,
# )

# model = EfficientAd(
#     imagenet_dir=r"D:\project computer vision\datasets\imagenette2-320\train"
# )

# engine = Engine(
#     default_root_dir=r"D:\project computer vision\outputs\checkpoint_eval",
#     accelerator="gpu",
#     devices=1,
# )

# engine.test(
#     model=model,
#     datamodule=datamodule,
#     ckpt_path=CKPT_PATH,
# )









# import torch
# from anomalib.data import MVTecLOCO
# from anomalib.models import EfficientAd
# from anomalib.engine import Engine

# torch.set_float32_matmul_precision("high")

# CKPT_PATH = r"D:\project computer vision\outputs\efficientad\EfficientAd\MVTecLOCO\splicing_connectors\v21\weights\lightning\model.ckpt"
# # أو إذا عملت copy:
# # CKPT_PATH = r"D:\project computer vision\outputs\efficientad\best_v21_model.ckpt"

# datamodule = MVTecLOCO(
#     root=r"D:\project computer vision\data",
#     category="splicing_connectors",
#     train_batch_size=1,
#     eval_batch_size=4,
#     num_workers=0,
# )

# model = EfficientAd(
#     imagenet_dir=r"D:\project computer vision\datasets\imagenette2-320\train"
# )

# engine = Engine(
#     default_root_dir=r"D:\project computer vision\outputs\checkpoint_eval",
#     accelerator="gpu",
#     devices=1,
# )

# engine.test(
#     model=model,
#     datamodule=datamodule,
#     ckpt_path=CKPT_PATH,
# )













# import torch
# from anomalib.data import MVTecLOCO
# from anomalib.models import EfficientAd
# from anomalib.engine import Engine

# torch.set_float32_matmul_precision("high")

# # CKPT_PATH = r"D:\project computer vision\outputs\efficientad_roi\EfficientAd\MVTecLOCO\splicing_connectors\v0\weights\lightning\model.ckpt"
# # # بعد أول run، بدّل v0 حسب الفولدر الفعلي اللي انخلق عندك

# CKPT_PATH = r"D:\project computer vision\outputs\efficientad_roi\best_efficientad_roi_v1.ckpt"
# root=r"D:\project computer vision\data_roi"


# datamodule = MVTecLOCO(
#     root=r"D:\project computer vision\data_roi",
#     category="splicing_connectors",
#     train_batch_size=1,
#     eval_batch_size=4,
#     num_workers=0,
# )

# model = EfficientAd(
#     imagenet_dir=r"D:\project computer vision\datasets\imagenette2-320\train"
# )

# engine = Engine(
#     default_root_dir=r"D:\project computer vision\outputs\checkpoint_eval_roi",
#     accelerator="gpu",
#     devices=1,
# )

# engine.test(
#     model=model,
#     datamodule=datamodule,
#     ckpt_path=CKPT_PATH,
# )









# import torch
# from anomalib.data import MVTecLOCO
# from anomalib.models import EfficientAd
# from anomalib.engine import Engine

# torch.set_float32_matmul_precision("high")

# CKPT_PATH = r"D:\project computer vision\outputs\efficientad_roi_bg\EfficientAd\MVTecLOCO\splicing_connectors\v1\weights\lightning\model.ckpt"

# datamodule = MVTecLOCO(
#     root=r"D:\project computer vision\data_roi_bg",
#     category="splicing_connectors",
#     train_batch_size=1,
#     eval_batch_size=4,
#     num_workers=0,
# )

# model = EfficientAd(
#     imagenet_dir=r"D:\project computer vision\datasets\imagenette2-320\train"
# )

# engine = Engine(
#     default_root_dir=r"D:\project computer vision\outputs\checkpoint_eval_roi_bg",
#     accelerator="gpu",
#     devices=1,
# )

# engine.test(
#     model=model,
#     datamodule=datamodule,
#     ckpt_path=CKPT_PATH,
# )


# import torch
# from anomalib.data import MVTecLOCO
# from anomalib.models import EfficientAd
# from anomalib.engine import Engine

# torch.set_float32_matmul_precision("high")

# CKPT_PATH = r"D:\project computer vision\outputs\efficientad_roi_bg_aug\best_efficientad_roi_bg_aug_v0.ckpt"

# datamodule = MVTecLOCO(
#     root=r"D:\project computer vision\data_roi_bg_aug",
#     category="splicing_connectors",
#     train_batch_size=1,
#     eval_batch_size=4,
#     num_workers=0,
# )

# model = EfficientAd(
#     imagenet_dir=r"D:\project computer vision\datasets\imagenette2-320\train"
# )

# engine = Engine(
#     default_root_dir=r"D:\project computer vision\outputs\checkpoint_eval_roi_bg_aug",
#     accelerator="gpu",
#     devices=1,
# )

# engine.test(
#     model=model,
#     datamodule=datamodule,
#     ckpt_path=CKPT_PATH,
# )





















import torch
from anomalib.data import MVTecLOCO
from anomalib.models import EfficientAd
from anomalib.engine import Engine

torch.set_float32_matmul_precision("high")

CKPT_PATH = r"D:\project computer vision\outputs\efficientad_roi_bg_aug\best_efficientad_roi_bg_aug_v0.ckpt"

datamodule = MVTecLOCO(
    root=r"D:\project computer vision\data_roi_bg_aug",
    category="splicing_connectors",
    train_batch_size=1,
    eval_batch_size=4,
    num_workers=0,
)

model = EfficientAd(
    imagenet_dir=r"D:\project computer vision\datasets\imagenette2-320\train"
)

engine = Engine(
    default_root_dir=r"D:\project computer vision\outputs\checkpoint_eval_roi_bg_aug",
    accelerator="gpu",
    devices=1,
)

engine.test(
    model=model,
    datamodule=datamodule,
    ckpt_path=CKPT_PATH,
)