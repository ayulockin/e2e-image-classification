import cv2
import numpy as np
import tensorflow as tf

import wandb
from wandb.keras import WandbEvalCallback


class WandbClfEvalCallback(WandbEvalCallback):
    def __init__(
        self, args, validloader, data_table_columns, pred_table_columns, num_samples=100, log_explainability=False
    ):
        self.args = args
        self.val_data = validloader.unbatch().take(num_samples)
        self.log_explainability = log_explainability

        if self.log_explainability:
            from tf_explain.core.grad_cam import GradCAM
            self.explainer = GradCAM()
            pred_table_columns.append("GradCAM")

        super().__init__(data_table_columns, pred_table_columns)

    def add_ground_truth(self, logs=None):
        for idx, (image, label) in enumerate(self.val_data):
            self.data_table.add_data(
                idx,
                wandb.Image(image),
                np.argmax(label, axis=-1)
            )

    def add_model_predictions(self, epoch, logs=None):
        # Get predictions
        preds = self._inference()
        table_idxs = self.data_table_ref.get_index()

        if self.log_explainability:
            heatmaps = self._gradcam_explain()

        for idx in table_idxs:
            pred = preds[idx]

            data = [
                epoch,
                self.data_table_ref.data[idx][0],
                self.data_table_ref.data[idx][1],
                self.data_table_ref.data[idx][2],
                pred
            ]

            if self.log_explainability:
                data.append(wandb.Image(heatmaps[idx]))

            self.pred_table.add_data(*data)

    def _inference(self):
        preds = []
        for image, label in self.val_data:
            pred = self.model(tf.expand_dims(image, axis=0))
            argmax_pred = tf.argmax(pred, axis=-1).numpy()[0]
            preds.append(argmax_pred)

        return preds

    def _gradcam_explain(self):
        heatmaps = []
        for image, label in self.val_data:
            image = tf.expand_dims(image, axis=0).numpy()
            label = np.argmax(label) if self.args.one_hot else label.numpy()

            heatmap = self.explainer.explain(
                validation_data=(image, label),
                model=self.model,
                class_index=label, # class index for the ground truth label (we can experiment with predicted label too)
                layer_name=None,
                use_guided_grads=True,
                colormap=cv2.COLORMAP_VIRIDIS,
                image_weight=0.7,
            )
            heatmaps.append(heatmap)

        return heatmaps
