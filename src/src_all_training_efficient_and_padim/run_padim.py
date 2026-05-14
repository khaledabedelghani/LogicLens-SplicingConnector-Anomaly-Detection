import torch
from multiprocessing import freeze_support

from anomalib.data import MVTecLOCO
from anomalib.models import Padim
from anomalib.engine import Engine


def main():
    torch.set_float32_matmul_precision("high")

    datamodule = MVTecLOCO(
        root=r"D:\project computer vision\data_roi_padim_aug",
        category="splicing_connectors",
        train_batch_size=8,
        eval_batch_size=4,
        num_workers=8,
    )

    model = Padim(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
        pre_trained=True,
        n_features=None,
    )

    engine = Engine(
        default_root_dir=r"D:\project computer vision\outputs\padim_roi_padim_aug",
        accelerator="gpu",
        devices=1,
        max_epochs=15,
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