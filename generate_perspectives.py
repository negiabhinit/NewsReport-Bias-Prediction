import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
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

def load_model_and_tokenizer(model_path):
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = get_device()
    model.to(device)
    return model, tokenizer, device

def generate_alternative_perspective(text, model, tokenizer, device):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    
    # Map prediction to perspective
    perspective_map = {
        0: "Neutral",
        1: "Left-leaning",
        2: "Right-leaning"
    }
    
    return {
        'predicted_perspective': perspective_map[pred_class],
        'confidence': confidence,
        'probabilities': probs[0].cpu().numpy()
    }

def generate_alternative_perspectives(original_statement):
    # Generate different perspectives on the same statement
    perspectives = {
        "Original Statement": original_statement,
        
        "Left-leaning Alternative": f"""
        {original_statement} However, this growth has not been evenly distributed, 
        with wealth inequality reaching historic levels. While corporate profits 
        and stock markets are soaring, many working-class families continue to 
        struggle with rising costs of living and stagnant wages. The current 
        economic policies favor large corporations and the wealthy, while failing 
        to address systemic issues like affordable healthcare, student debt, and 
        the growing wealth gap. Progressive policies are needed to ensure that 
        economic growth benefits all Americans, not just the top 1%.
        """,
        
        "Neutral Alternative": f"""
        {original_statement} Economic indicators show positive trends in several 
        key areas, including GDP growth, employment rates, and consumer spending. 
        However, challenges remain in certain sectors and regions. The economy's 
        performance varies across different industries and demographic groups, 
        with some experiencing significant growth while others face ongoing 
        difficulties. A comprehensive analysis of economic data reveals both 
        strengths and areas that require attention for sustainable growth.
        """,
        
        "Right-leaning Alternative": f"""
        {original_statement} This success is a direct result of pro-business 
        policies, tax cuts, and deregulation that have unleashed the power of 
        the free market. The private sector's innovation and entrepreneurship 
        have driven job creation and economic expansion. Government intervention 
        should be minimized to allow businesses to thrive and continue creating 
        opportunities for all Americans. The current economic policies have 
        proven that free-market principles lead to prosperity and should be 
        maintained and expanded.
        """
    }
    return perspectives

def analyze_perspectives(perspectives, model, tokenizer, device):
    results = []
    for perspective_name, text in perspectives.items():
        result = generate_alternative_perspective(text, model, tokenizer, device)
        result['perspective_name'] = perspective_name
        results.append(result)
        
        # Print results
        logger.info(f"\nAnalyzing {perspective_name}:")
        logger.info(f"Predicted Perspective: {result['predicted_perspective']}")
        logger.info(f"Confidence: {result['confidence']:.2%}")
        logger.info("Probability Distribution:")
        logger.info(f"Neutral: {result['probabilities'][0]:.2%}")
        logger.info(f"Left-leaning: {result['probabilities'][1]:.2%}")
        logger.info(f"Right-leaning: {result['probabilities'][2]:.2%}")
    
    return results

def main():
    try:
        # Load model and tokenizer
        model, tokenizer, device = load_model_and_tokenizer('./model')
        
        # Original statement
        original_statement = "United states economy is doing well."
        
        # Generate alternative perspectives
        logger.info("Generating alternative perspectives...")
        perspectives = generate_alternative_perspectives(original_statement)
        
        # Analyze perspectives
        results = analyze_perspectives(perspectives, model, tokenizer, device)
        
        # Save results
        pd.DataFrame(results).to_csv('economy_perspectives_analysis.csv', index=False)
        logger.info("\nResults saved to economy_perspectives_analysis.csv")
        
        # Print the generated perspectives
        logger.info("\nGenerated Perspectives:")
        for name, text in perspectives.items():
            logger.info(f"\n{name}:")
            logger.info(text.strip())
        
    except Exception as e:
        logger.error(f"Error in perspective generation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 