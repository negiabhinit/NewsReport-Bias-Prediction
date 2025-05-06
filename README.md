# News Report Bias Prediction and New perspective Generation

This project focuses on analyzing political bias in news articles and generating alternative perspectives using machine learning models. It combines classification and text generation capabilities to provide insights into media bias and create balanced viewpoints.

## Project Structure

### Core Files

1. `data_preparation.py`
   - Prepares and processes training data
   - Handles data splitting (media-based and random splits)
   - Merges datasets and saves them in appropriate formats

2. `model_training.py`
   - Implements the training pipeline for the bias classification model
   - Uses BERT-based architecture for sequence classification
   - Includes evaluation metrics and model saving functionality

3. `generate_perspectives.py`
   - Basic version for generating and analyzing perspectives
   - Uses the trained classification model for bias analysis

4. `generate_perspectives_advanced.py`
   - Advanced version combining classification and generation
   - Uses both BERT (for classification) and GPT-Neo (for text generation)
   - Provides detailed analysis of input and generated perspectives

### Data Files

- `data/splits/media/`: Contains media-based split datasets
- `data/splits/random/`: Contains random split datasets
- `model_media/`: Saved model trained on media-based split
- `model_random/`: Saved model trained on random split

### Output Files

- `perspective_analysis.csv`: Stores results of perspective generation and analysis
- `economy_perspectives_analysis.csv`: Specific analysis for economic perspectives

## Setup and Installation

1. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preparation:
```bash
python data_preparation.py
```

2. Model Training:
```bash
python model_training.py
```

3. Generate Perspectives:
```bash
python generate_perspectives_advanced.py
```

## Model Architecture

- Classification Model: BERT-based sequence classification
- Generation Model: GPT-Neo for text generation
- Training: 15 epochs with detailed metrics tracking
- Evaluation: Per-class precision, recall, and F1 scores

## Features

- Political bias classification (Left-leaning, Right-leaning, Neutral)
- Alternative perspective generation
- Detailed confidence scores and probability distributions
- Support for both media-based and random data splits
- GPU acceleration support (CUDA and MPS)

## Requirements

- Python 3.8+
- PyTorch
- Transformers library
- Pandas
- NumPy
- tqdm

## Dataset

The project uses the following datasets and models:

1. **Original Dataset**: Articles crawled from www.allsides.com
   - Total of 37,554 articles
   - Available in JSON format in `./data/jsons`
   - Each article contains:
     - ID, topic, source, URL, date, authors
     - Title and content (original and processed)
     - Political bias annotation (left, center, right)

2. **Trained Models**:
   - Media-based split model: [Download Link](https://drive.google.com/file/d/1DknQhPeAcdZYhEJ-L5xkm4puJ1W2sE9E/view?usp=share_link)
   - Random split model: Available in `model_random/` directory

## References

1. Original Paper:
   - Baly, R., Da San Martino, G., Glass, J., & Nakov, P. (2020). We Can Detect Your Bias: Predicting the Political Ideology of News Articles. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 4982-4991.
   - [Paper Link](https://aclanthology.org/2020.emnlp-main.404.pdf)

2. Related Projects:
   - [Original Implementation](https://github.com/ramybaly/Article-Bias-Prediction)
   - [Qbias Project](https://github.com/irgroup/Qbias)

## Notes

- The project uses two different data splitting strategies:
  - Media-based split: Maintains context and source relationships
  - Random split: Traditional random distribution
- Models are saved in separate directories for different training approaches
- The advanced perspective generator combines classification and generation capabilities

## Dataset
The articles crawled from www.allsides.com are available in the ```./data``` folder, along with the different evaluation splits.

The dataset consists of a total of 37,554 articles. Each article is stored as a ```JSON``` object in the ```./data/jsons``` directory, and contains the following fields:
1. **ID**: an alphanumeric identifier.
2. **topic**: the topic being discussed in the article.
3. **source**: the name of the articles's source *(example: New York Times)*
4. **source_url**: the URL to the source's homepage *(example: www.nytimes.com)*
5. **url**: the link to the actual article.
6. **date**: the publication date of the article.
7. **authors**: a comma-separated list of the article's authors.
8. **title**: the article's title.
9. **content_original**: the original body of the article, as returned by the ```newspaper3k``` Python library.
10. **content**: the processed and tokenized content, which is used as input to the different models.
11. **bias_text**: the label of the political bias annotation of the article (left, center, or right).
12. **bias**: the numeric encoding of the political bias of the article (0, 1, or 2).

The ```./data/splits``` directory contains the two types of splits, as discussed in the paper: **random** and **media-based**. For each of these types, we provide the train, validation and test files that contains the articles' IDs belonging to each set, along with their numeric bias label.

## Citation

```
@inproceedings{baly2020we,
  author      = {Baly, Ramy and Da San Martino, Giovanni and Glass, James and Nakov, Preslav},
  title       = {We Can Detect Your Bias: Predicting the Political Ideology of News Articles},
  booktitle   = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  series      = {EMNLP~'20},
  NOmonth     = {November},
  year        = {2020}
  pages       = {4982--4991},
  NOpublisher = {Association for Computational Linguistics}
}
```
# NewsReport-Bias-Prediction
