import pandas as pd
import json
import os
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
splits_dir = 'data/splits/media'
jsons_dir = 'data/jsons'

def load_json_data(json_dir, id_list):
    data = []
    missing_files = []
    for id in tqdm(id_list, desc="Loading JSON files"):
        json_path = os.path.join(json_dir, f'{id}.json')
        try:
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                    # Validate required fields
                    required_fields = ['content', 'topic']
                    if all(field in json_data for field in required_fields):
                        data.append({
                            'ID': id,
                            'content': json_data['content'],
                            'topic': json_data['topic']
                        })
                    else:
                        logger.warning(f"Missing required fields in {id}.json")
            else:
                missing_files.append(id)
        except Exception as e:
            logger.error(f"Error processing {id}.json: {str(e)}")
            continue
    
    if missing_files:
        logger.warning(f"Missing {len(missing_files)} JSON files")
    return data

def main():
    try:
        # Read split files
        logger.info("Reading split files...")
        train_df = pd.read_csv(os.path.join(splits_dir, 'train.tsv'), sep='\t')
        valid_df = pd.read_csv(os.path.join(splits_dir, 'valid.tsv'), sep='\t')
        test_df = pd.read_csv(os.path.join(splits_dir, 'test.tsv'), sep='\t')

        # Load JSON data for each split
        logger.info("Loading JSON data...")
        train_data = load_json_data(jsons_dir, train_df['ID'].tolist())
        valid_data = load_json_data(jsons_dir, valid_df['ID'].tolist())
        test_data = load_json_data(jsons_dir, test_df['ID'].tolist())

        # Convert to DataFrame
        logger.info("Converting to DataFrames...")
        train_json_df = pd.DataFrame(train_data)
        valid_json_df = pd.DataFrame(valid_data)
        test_json_df = pd.DataFrame(test_data)

        # Merge with split DataFrames
        logger.info("Merging datasets...")
        train_merged = pd.merge(train_df, train_json_df, on='ID')
        valid_merged = pd.merge(valid_df, valid_json_df, on='ID')
        test_merged = pd.merge(test_df, test_json_df, on='ID')

        # Save merged DataFrames
        logger.info("Saving merged datasets...")
        train_merged.to_csv('media_train_merged.csv', index=False)
        valid_merged.to_csv('media_valid_merged.csv', index=False)
        test_merged.to_csv('media_test_merged.csv', index=False)

        logger.info("Data preparation completed successfully!")
        logger.info(f"Train samples: {len(train_merged)}")
        logger.info(f"Validation samples: {len(valid_merged)}")
        logger.info(f"Test samples: {len(test_merged)}")

    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 