import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPTNeoForCausalLM
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

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

def load_models():
    logger.info("Loading models...")
    device = get_device()
    
    # Load classification model
    classifier_tokenizer = AutoTokenizer.from_pretrained('./model_random')
    classifier_model = AutoModelForSequenceClassification.from_pretrained('./model_random')
    classifier_model.to(device)
    
    # Load text generation model
    logger.info("Loading GPT-Neo model...")
    generator_tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    generator_model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
    generator_model.to(device)
    
    return classifier_model, classifier_tokenizer, generator_model, generator_tokenizer, device

def classify_perspective(text, model, tokenizer, device):
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

def generate_alternative_perspective(input_text, generator_model, generator_tokenizer, device):
    # Use input text directly as prompt
    inputs = generator_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=100)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = generator_model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=generator_tokenizer.eos_token_id,
            repetition_penalty=1.5,
            no_repeat_ngram_size=4,
            early_stopping=True
        )
    
    # Decode and clean the generated text
    generated_text = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the input text if it appears at the start
    if generated_text.startswith(input_text):
        generated_text = generated_text[len(input_text):].strip()
    
    return generated_text

def main():
    try:
        # Load models
        classifier_model, classifier_tokenizer, generator_model, generator_tokenizer, device = load_models()
        
        # Get input from user
        input_text = input("\nEnter your perspective or statement: ").strip()
        
        # Analyze the input
        logger.info("\nAnalyzing input perspective...")
        classification = classify_perspective(input_text, classifier_model, classifier_tokenizer, device)
        
        # Print analysis results
        logger.info(f"\nInput Analysis:")
        logger.info(f"Original Text: \"{input_text}\"")
        logger.info(f"Predicted Perspective: {classification['predicted_perspective']}")
        logger.info(f"Confidence: {classification['confidence']:.2%}")
        logger.info("Probability Distribution:")
        logger.info(f"Neutral: {classification['probabilities'][0]:.2%}")
        logger.info(f"Left-leaning: {classification['probabilities'][1]:.2%}")
        logger.info(f"Right-leaning: {classification['probabilities'][2]:.2%}")
        
        # Generate alternative perspective
        logger.info("\nGenerating alternative perspective...")
        alternative_perspective = generate_alternative_perspective(
            input_text,
            generator_model,
            generator_tokenizer,
            device
        )
        
        # Classify the alternative perspective
        alt_classification = classify_perspective(
            alternative_perspective,
            classifier_model,
            classifier_tokenizer,
            device
        )
        
        # Print alternative perspective results
        logger.info(f"\nAlternative Perspective:")
        logger.info(f"\"{alternative_perspective}\"")
        logger.info(f"\nAlternative Perspective Analysis:")
        logger.info(f"Predicted Perspective: {alt_classification['predicted_perspective']}")
        logger.info(f"Confidence: {alt_classification['confidence']:.2%}")
        logger.info("Probability Distribution:")
        logger.info(f"Neutral: {alt_classification['probabilities'][0]:.2%}")
        logger.info(f"Left-leaning: {alt_classification['probabilities'][1]:.2%}")
        logger.info(f"Right-leaning: {alt_classification['probabilities'][2]:.2%}")
        
        # Save results
        results = [
            {
                'text': input_text,
                'perspective': classification['predicted_perspective'],
                'confidence': classification['confidence'],
                'type': 'original'
            },
            {
                'text': alternative_perspective,
                'perspective': alt_classification['predicted_perspective'],
                'confidence': alt_classification['confidence'],
                'type': 'alternative'
            }
        ]
        
        df = pd.DataFrame(results)
        df.to_csv('perspective_analysis.csv', index=False)
        logger.info("\nResults saved to perspective_analysis.csv")
        
    except Exception as e:
        logger.error(f"Error in perspective generation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 