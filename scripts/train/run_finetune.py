from dataclasses import dataclass, field
from typing import Optional, Union
import os

import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error
# train-test split
from sklearn.model_selection import train_test_split
import torch

from transformers import TrainingArguments, Trainer, HfArgumentParser, TrainerCallback, TrainerControl, TrainerState

from rnalens.model import BertForSequenceClassification, BertConfig
from rnalens.data import FinetuneDataset, Alphabet
from transformers import AutoTokenizer

@dataclass
class FinetuneArguments:
    """
    Arguments for pretraining the rnalens model.
    """
    data_fpath: str = field(
        metadata={"help": "Path to the training data file. should be ended with `.csv`."}
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to the pretrained model."}
    )
    test_data_fpath: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the test data file. should be ended with `.csv`. If not provided we would use train-test split."}
    )
    seq_type: str = field(
        default="utr",
        metadata={"help": "Type of sequence to be used for training."}
    )
    label_type: str = field(
        default="rnaseq_log", # te_log for te-tasks
        metadata={"help": "Type of label to be used for training."}
    )
    tokenizer_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to the tokenizer."}
    )

from transformers import TrainerCallback, TrainerControl, TrainerState, Trainer

class SpearmanEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int = 5, threshold: float = 1e-4):
        self.patience = patience
        self.threshold = threshold
        self.best_score = None
        self.counter = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        score = metrics.get("eval_Spearman")

        if score is None:
            print("[warn] Spearman not found in metrics")
            return control

        if self.best_score is None or score > self.best_score + self.threshold:
            self.best_score = score
            self.counter = 0
            control.should_save = True
        else:
            self.counter += 1
            print(f"[early stop] Spearman hasn't improved for {self.counter} evals.")
            if self.counter >= self.patience:
                control.should_training_stop = True

        return control


def compute_metrics(eval_pred):
    """
    Compute metrics used for huggingface trainer.
    """
    predictions, labels = eval_pred
    
    predictions.reshape(-1)
    labels.reshape(-1)
    
    return {
        "MSE": mean_squared_error(labels, predictions),
        "MAE": mean_absolute_error(labels, predictions),
        "R2": r2_score(labels, predictions),
        "Spearman": spearmanr(labels, predictions).correlation,
        "max_y": np.max(labels),
        "min_y": np.min(labels),
        "max_pred": np.max(predictions),
        "min_pred": np.min(predictions),
    }

def preprocess_logits_for_metrics(logits: Union[torch.Tensor, tuple[torch.Tensor, any]], _):
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    return logits

def save_pred_results(
    trainer,
    dataset,
    out_path,
):
    # save prediction results
    preds = trainer.predict(dataset).predictions
    
    results = {
        "text": [],
        "actual_labels": [],
        "prediction_labels": [],
    }
    for i, data in enumerate(dataset):
        results["text"].append(data["sequence_str"])
        results["actual_labels"].append(data["labels"].item())
        results["prediction_labels"].append(preds[i][0])

    train_df = pd.DataFrame(results)
    train_df.to_csv(out_path)

if __name__ == "__main__":
    parser = HfArgumentParser((FinetuneArguments, TrainingArguments))
    finetune_args, training_args = parser.parse_args_into_dataclasses()
    print("[info] per_device_train_batch_size=", training_args.per_device_train_batch_size)

    if finetune_args.tokenizer_name_or_path is None:
        raise ValueError("tokenizer_name_or_path must be provided")
    
    print(f"[info] Loading tokenizer {finetune_args.tokenizer_name_or_path}...")
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(finetune_args.tokenizer_name_or_path)

    csv_data = pd.read_csv(finetune_args.data_fpath)

    if finetune_args.test_data_fpath is None:
        csv_data_train, csv_data_test = train_test_split(
            csv_data,
            test_size=0.1,
            random_state=42,
            shuffle=True,
        )
    else:
        csv_data_train = csv_data
        csv_data_test = pd.read_csv(
            finetune_args.test_data_fpath
        )

    # Load the dataset
    print("[info] Loading dataset...")
    train_dataset = FinetuneDataset(
        csv_data_train.loc[:, finetune_args.label_type],
        csv_data_train[finetune_args.seq_type],
        mask_prob=0,
        tokenizer=tokenizer,
        clean_seq=True, # TODO: expose this to arguments
    )
    test_dataset = FinetuneDataset(
        csv_data_test.loc[:, finetune_args.label_type],
        csv_data_test[finetune_args.seq_type],
        mask_prob=0,
        tokenizer=tokenizer,
        clean_seq=True, # TODO: expose this to arguments
    )

    print("[info] Loading model...")
    config = BertConfig.from_pretrained(finetune_args.model_name_or_path)
    config.problem_type = "regression"
    config.num_labels = 1

    model = BertForSequenceClassification.from_pretrained(
        finetune_args.model_name_or_path,
        config=config
    )

    # setup the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,  # Yes, the original paper is actually using the test set for evaluation
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        #callbacks=[SpearmanEarlyStoppingCallback(patience=20)]
    )

    trainer.train()

    # eval
    results = trainer.evaluate(test_dataset)
    with open(os.path.join(training_args.output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("[info] Evaluation results saved to eval_results.json")
    
    
    # save prediction results
    save_pred_results(
        trainer,
        dataset=train_dataset,
        out_path=os.path.join(training_args.output_dir, "pred_results_train.csv"),
    )
    print("[info] Prediction results saved to ", os.path.join(training_args.output_dir, "pred_results_train.csv"))

    save_pred_results(
        trainer,
        dataset=test_dataset,
        out_path=os.path.join(training_args.output_dir, "pred_results_test.csv"),
    )
    print("[info] Prediction results saved to ", os.path.join(training_args.output_dir, "pred_results_test.csv"))
