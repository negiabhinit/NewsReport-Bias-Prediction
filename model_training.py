import pandas as pd
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA (GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple GPU)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device

def validate_data(df, name):
    required_columns = ['content', 'bias']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns in {name} dataset")
    if df['content'].isna().any():
        logger.warning(f"Found {df['content'].isna().sum()} missing values in {name} content")
        df = df.dropna(subset=['content'])
    if df['bias'].isna().any():
        logger.warning(f"Found {df['bias'].isna().sum()} missing values in {name} bias")
        df = df.dropna(subset=['bias'])
    return df

def main():
    try:
        # Load prepared datasets
        logger.info("Loading datasets...")
        train_df = pd.read_csv('media_train_merged.csv')
        valid_df = pd.read_csv('media_valid_merged.csv')
        test_df = pd.read_csv('media_test_merged.csv')

        # Validate data
        logger.info("Validating datasets...")
        train_df = validate_data(train_df, 'train')
        valid_df = validate_data(valid_df, 'valid')
        test_df = validate_data(test_df, 'test')

        # Convert to HuggingFace Dataset
        logger.info("Converting to HuggingFace datasets...")
        train_dataset = Dataset.from_pandas(train_df)
        valid_dataset = Dataset.from_pandas(valid_df)
        test_dataset = Dataset.from_pandas(test_df)

        # Initialize tokenizer and model
        logger.info("Initializing model and tokenizer...")
        model_name = 'bert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        # Preprocess function
        def preprocess_function(examples):
            tokenized = tokenizer(examples['content'], truncation=True, padding='max_length', max_length=128)
            tokenized['labels'] = examples['bias']
            return tokenized

        # Apply preprocessing
        logger.info("Preprocessing datasets...")
        encoded_train = train_dataset.map(preprocess_function, batched=True)
        encoded_valid = valid_dataset.map(preprocess_function, batched=True)
        encoded_test = test_dataset.map(preprocess_function, batched=True)

        # Set format for PyTorch
        encoded_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        encoded_valid.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        encoded_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=15,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            weight_decay=0.01,
            logging_dir='./logs',
            save_total_limit=2,
            logging_steps=100,
            save_steps=500,
        )

        # Compute metrics
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=1)
            acc = np.mean(preds == labels)
            
            # Calculate per-class metrics
            unique_labels = np.unique(labels)
            metrics = {'accuracy': acc}
            
            for label in unique_labels:
                label_mask = labels == label
                pred_mask = preds == label
                precision = np.sum((preds == label) & (labels == label)) / np.sum(preds == label) if np.sum(preds == label) > 0 else 0
                recall = np.sum((preds == label) & (labels == label)) / np.sum(labels == label) if np.sum(labels == label) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics.update({
                    f'precision_class_{label}': precision,
                    f'recall_class_{label}': recall,
                    f'f1_class_{label}': f1
                })
            
            return metrics

        # Initialize Trainer
        device = get_device()
        model.to(device)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_train,
            eval_dataset=encoded_valid,
            compute_metrics=compute_metrics,
        )

        # Train the model
        logger.info("Starting training...")
        trainer.train()

        # Evaluate the model
        logger.info("Evaluating model...")
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")

        # Predict on test set
        logger.info("Making predictions on test set...")
        preds = trainer.predict(encoded_test)
        pred_labels = np.argmax(preds.predictions, axis=1)
        true_labels = preds.label_ids
        logger.info("Classification Report:")
        logger.info(classification_report(true_labels, pred_labels, target_names=['0', '1', '2']))

        # Save the model
        logger.info("Saving model...")
        model.save_pretrained('./model')
        tokenizer.save_pretrained('./model')
        logger.info("Model saved successfully!")

    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 