import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
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

def load_model_and_tokenizer(model_path):
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = get_device()
    model.to(device)
    return model, tokenizer, device

def analyze_class_distribution(df, split_name):
    logger.info(f"\nClass distribution in {split_name} set:")
    class_counts = df['bias'].value_counts()
    class_percentages = df['bias'].value_counts(normalize=True) * 100
    
    for bias_class, count in class_counts.items():
        percentage = class_percentages[bias_class]
        logger.info(f"Class {bias_class}: {count} samples ({percentage:.2f}%)")
    
    return class_counts, class_percentages

def plot_class_distribution(class_counts, title, filename):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title(title)
    plt.xlabel('Bias Class')
    plt.ylabel('Number of Samples')
    plt.savefig(filename)
    plt.close()

def analyze_text_lengths(df, split_name):
    logger.info(f"\nText length analysis for {split_name} set:")
    df['text_length'] = df['content'].str.len()
    
    logger.info(f"Average text length: {df['text_length'].mean():.2f} characters")
    logger.info(f"Minimum text length: {df['text_length'].min()} characters")
    logger.info(f"Maximum text length: {df['text_length'].max()} characters")
    
    return df['text_length']

def plot_text_length_distribution(lengths, title, filename):
    plt.figure(figsize=(10, 6))
    sns.histplot(lengths, bins=50)
    plt.title(title)
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Number of Samples')
    plt.savefig(filename)
    plt.close()

def analyze_model_predictions(model, tokenizer, device, test_df):
    logger.info("\nAnalyzing model predictions...")
    
    predictions = []
    true_labels = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        inputs = tokenizer(row['content'], return_tensors="pt", truncation=True, padding='max_length', max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        
        predictions.append(pred)
        true_labels.append(row['bias'])
    
    return np.array(predictions), np.array(true_labels)

def plot_confusion_matrix(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(filename)
    plt.close()

def main():
    try:
        # Load data
        logger.info("Loading datasets...")
        train_df = pd.read_csv('train_merged.csv')
        valid_df = pd.read_csv('valid_merged.csv')
        test_df = pd.read_csv('test_merged.csv')
        
        # Analyze class distribution
        train_counts, train_percentages = analyze_class_distribution(train_df, 'Training')
        valid_counts, valid_percentages = analyze_class_distribution(valid_df, 'Validation')
        test_counts, test_percentages = analyze_class_distribution(test_df, 'Test')
        
        # Plot class distributions
        plot_class_distribution(train_counts, 'Training Set Class Distribution', 'train_class_dist.png')
        plot_class_distribution(valid_counts, 'Validation Set Class Distribution', 'valid_class_dist.png')
        plot_class_distribution(test_counts, 'Test Set Class Distribution', 'test_class_dist.png')
        
        # Analyze text lengths
        train_lengths = analyze_text_lengths(train_df, 'Training')
        valid_lengths = analyze_text_lengths(valid_df, 'Validation')
        test_lengths = analyze_text_lengths(test_df, 'Test')
        
        # Plot text length distributions
        plot_text_length_distribution(train_lengths, 'Training Set Text Length Distribution', 'train_length_dist.png')
        plot_text_length_distribution(valid_lengths, 'Validation Set Text Length Distribution', 'valid_length_dist.png')
        plot_text_length_distribution(test_lengths, 'Test Set Text Length Distribution', 'test_length_dist.png')
        
        # Load model and analyze predictions
        model, tokenizer, device = load_model_and_tokenizer('./model')
        predictions, true_labels = analyze_model_predictions(model, tokenizer, device, test_df)
        
        # Plot confusion matrix
        plot_confusion_matrix(true_labels, predictions, 'confusion_matrix.png')
        
        # Print classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(true_labels, predictions, target_names=['Neutral', 'Left-leaning', 'Right-leaning']))
        
        logger.info("Analysis completed successfully!")
        logger.info("Results saved in the following files:")
        logger.info("- train_class_dist.png")
        logger.info("- valid_class_dist.png")
        logger.info("- test_class_dist.png")
        logger.info("- train_length_dist.png")
        logger.info("- valid_length_dist.png")
        logger.info("- test_length_dist.png")
        logger.info("- confusion_matrix.png")
        
    except Exception as e:
        logger.error(f"Error in performance analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 