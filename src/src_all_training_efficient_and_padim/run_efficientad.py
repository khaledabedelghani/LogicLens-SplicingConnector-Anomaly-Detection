# import torch
# from anomalib.data import MVTecLOCO
# from anomalib.models import EfficientAd
# from anomalib.engine import Engine

# torch.set_float32_matmul_precision("medium")

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
#     default_root_dir=r"D:\project computer vision\outputs\efficientad",
#     accelerator="gpu",
#     devices=1,
# )

# engine.fit(
#     model=model,
#     datamodule=datamodule,
#     ckpt_path=r"D:\project computer vision\outputs\efficientad\EfficientAd\MVTecLOCO\splicing_connectors\v20\weights\lightning\model.ckpt"
# )

# engine.test(model=model, datamodule=datamodule)



















# import torch
# from multiprocessing import freeze_support
# from anomalib.data import MVTecLOCO
# from anomalib.models import EfficientAd
# from anomalib.engine import Engine


# def main():
#     torch.set_float32_matmul_precision("high")

#     datamodule = MVTecLOCO(
#         root=r"D:\project computer vision\data",
#         category="splicing_connectors",
#         train_batch_size=1,
#         eval_batch_size=4,
#         num_workers=0,
#     )

#     model = EfficientAd(
#         imagenet_dir=r"D:\project computer vision\datasets\imagenette2-320\train"
#     )

#     engine = Engine(
#         default_root_dir=r"D:\project computer vision\outputs\efficientad",
#         accelerator="gpu",
#         devices=1,
#         precision="16-mixed",
#     )

#     engine.test(
#         model=model,
#         datamodule=datamodule,
#         ckpt_path=r"D:\project computer vision\outputs\efficientad\EfficientAd\MVTecLOCO\splicing_connectors\v20\weights\lightning\model.ckpt"
#     )


# if __name__ == "__main__":
#     freeze_support()
#     main()












# import torch
# from multiprocessing import freeze_support
# from anomalib.data import MVTecLOCO
# from anomalib.models import EfficientAd
# from anomalib.engine import Engine

# def main():
#     torch.set_float32_matmul_precision("high")

#     datamodule = MVTecLOCO(
#         root=r"D:\project computer vision\data",
#         category="splicing_connectors",
#         train_batch_size=1,
#         eval_batch_size=4,
#         num_workers=2,
#     )

#     model = EfficientAd(
#         imagenet_dir=r"D:\project computer vision\datasets\imagenette2-320\train"
#     )

#     engine = Engine(
#         default_root_dir=r"D:\project computer vision\outputs\efficientad",
#         accelerator="gpu",
#         devices=1,
#         precision="16-mixed",
#     )

#     engine.fit(
#         model=model,
#         datamodule=datamodule,
#         ckpt_path=r"D:\project computer vision\outputs\efficientad\EfficientAd\MVTecLOCO\splicing_connectors\v23\weights\lightning\model.ckpt"
#     )

#     engine.test(model=model, datamodule=datamodule)

# if __name__ == "__main__":
#     freeze_support()
#     main()










# import torch
# from multiprocessing import freeze_support
# from anomalib.data import MVTecLOCO
# from anomalib.models import EfficientAd
# from anomalib.engine import Engine

# def main():
#     torch.set_float32_matmul_precision("high")

#     datamodule = MVTecLOCO(
#         root=r"D:\project computer vision\data",
#         category="splicing_connectors",
#         train_batch_size=1,
#         eval_batch_size=4,
#         num_workers=2,
#     )

#     model = EfficientAd(
#         imagenet_dir=r"D:\project computer vision\datasets\imagenette2-320\train"
#     )

#     engine = Engine(
#         default_root_dir=r"D:\project computer vision\outputs\efficientad",
#         accelerator="gpu",
#         devices=1,
#         precision="16-mixed",
#     )

#     engine.fit(
#         model=model,
#         datamodule=datamodule,
#         ckpt_path=r"D:\project computer vision\outputs\efficientad\EfficientAd\MVTecLOCO\splicing_connectors\v21\weights\lightning\model.ckpt"
#     )

#     engine.test(model=model, datamodule=datamodule)

# if __name__ == "__main__":
#     freeze_support()
#     main()







# he la a7san we7ke te3et v1

# import torch
# from multiprocessing import freeze_support
# from anomalib.data import MVTecLOCO
# from anomalib.models import EfficientAd
# from anomalib.engine import Engine

# def main():
#     torch.set_float32_matmul_precision("high")

#     datamodule = MVTecLOCO(
#         root=r"D:\project computer vision\data_roi",
#         category="splicing_connectors",
#         train_batch_size=1,
#         eval_batch_size=4,
#         num_workers=2,
#     )

#     model = EfficientAd(
#         imagenet_dir=r"D:\project computer vision\datasets\imagenette2-320\train"
#     )

#     engine = Engine(
#     default_root_dir=r"D:\project computer vision\outputs\efficientad_roi",
#     accelerator="gpu",
#     devices=1,
#     precision="16-mixed",
#     max_epochs=150,
# )

#     engine.fit(
#         model=model,
#         datamodule=datamodule,
#     )

#     engine.test(
#         model=model,
#         datamodule=datamodule,
#     )

# if __name__ == "__main__":
#     freeze_support()
#     main()










import torch
from multiprocessing import freeze_support
from anomalib.data import MVTecLOCO
from anomalib.models import EfficientAd
from anomalib.engine import Engine

def main():
    torch.set_float32_matmul_precision("high")

    datamodule = MVTecLOCO(
        root = r"D:\project computer vision\data_roi_bg_aug",
        category="splicing_connectors",
        train_batch_size=1,
        eval_batch_size=4,
        num_workers=2,
    )

    model = EfficientAd(
        imagenet_dir=r"D:\project computer vision\datasets\imagenette2-320\train"
    )

    engine = Engine(
        default_root_dir = r"D:\project computer vision\outputs\efficientad_roi_bg_aug",
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=200,
    )

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    engine.test(
        model=model,
        datamodule=datamodule,
    )

if __name__ == "__main__":
    freeze_support()
    main()