import os
import yaml
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
import wandb
import tensorflow as tf

from img_clf.dataloader import GetDataloader
from img_clf.model import get_model

from wandb_addons.callbacks.keras import WandbGradCAMCallback


def get_args():
    parser = argparse.ArgumentParser(description="Train image classification model.")
    parser.add_argument(
        "--sweep_file", type=str, default="configs/sweeps.yaml", help="sweep file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs for training"
    )
    parser.add_argument(
        "--img_height", type=int, default=224, help="image height for training"
    )
    parser.add_argument(
        "--img_width", type=int, default=224, help="image width for training"
    )
    parser.add_argument(
        "--img_channels", type=int, default=3, help="image channels for training"
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=1000,
        help="shuffle buffer size for training",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="e2e-img-clf",
        help="wandb project name",
    )
    parser.add_argument(
        "--model_backbone", type=str, default="vgg16", help="backbone for the model"
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.2, help="Dropout rate post GAP"
    )
    parser.add_argument(
        "--num_classes", type=int, default=53, help="Number of classes in the dataset"
    )
    parser.add_argument(
        "--one_hot", type=bool, default=True, help="One hot encode the labels"
    )
    parser.add_argument(
        "--freeze_backbone", type=bool, default=True, help="Freeze the backbone layers"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for the optimizer"
    )

    return parser.parse_args()


def main(args: argparse.Namespace):
    # Get the dataloaders
    dataloader = GetDataloader(args)
    trainloader = dataloader.get_dataloader("train")
    validloader = dataloader.get_dataloader("val")

    # Get the id2label dict
    id2label = dataloader.get_id2label_dict()

    # Get the model
    model = get_model(args)
    model.summary()

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.CategoricalCrossentropy()
        if args.one_hot
        else tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # Initialize wandb
    wandb.init(project=args.wandb_project_name, entity="ml-colabs", config=vars(args))

    # Train the model
    model.fit(
        trainloader,
        epochs=args.epochs,
        validation_data=validloader,
        callbacks=[
            wandb.keras.WandbMetricsLogger(log_freq=2),
            WandbGradCAMCallback(
                validloader=validloader,
                data_table_columns=["idx", "image", "label"],
                pred_table_columns=["epoch", "idx", "image", "label", "pred"],
                one_hot_label=args.one_hot,
                id2label=id2label,
                log_explainability=True,
            ),
        ],
    )

    # Evaluate the model
    eval_loss, eval_acc = model.evaluate(validloader)
    wandb.log({"eval_loss": eval_loss, "eval_acc": eval_acc})

if __name__ == "__main__":
    args = get_args()
    print(args)

    main(args)
