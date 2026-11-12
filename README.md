# German Compound Splitting: NLP & Sequence Labeling

A Natural Language Processing (NLP) project focused on German compound word segmentation using character-level sequence labeling and machine learning.

The project explores how machine learning models can identify morphological boundaries within German compound words by analyzing character-level patterns and contextual information.

## Project Overview

German is well known for its productive compound word formation, where multiple words are combined to create a single complex word.

For example:

- `Donaudampfschifffahrt` → `Donau dampf schiff fahrt`
- `Computerlinguistik` → `Computer linguistik`

Automatically identifying these boundaries is an important NLP task with applications in tokenization, morphological analysis, and text processing.

**Objective:** Develop machine learning approaches to identify and predict segmentation boundaries in German compound words.

## Technical Skills & Technologies

- **Programming:** Python
- **Natural Language Processing:** German text processing, morphological segmentation
- **Machine Learning:** Sequence labeling, character-level classification
- **Feature Engineering:** Character-level contextual features
- **Sequence Labeling:** B/I tagging (Beginning / Inside)
- **Libraries:** Scikit-learn, NumPy, Pandas
- **Development:** Git, GitHub

## Methodology

The segmentation problem can be formulated as a character-level sequence labeling task.

Each character receives a label indicating whether it begins a new morphological unit or belongs to an existing unit.

- **B (Beginning):** Beginning of a morphological unit.
- **I (Inside):** Character inside a morphological unit.

For example:

```text
Input:  computerlinguistik
Labels: BIIIIIIIBIIIIIIIII
Output: computer linguistik
```

The predicted character labels are converted back into segmented compound words by inserting spaces at the identified boundaries.

## Machine Learning Approaches

The project explores supervised learning methods for character-level sequence prediction.

Potential approaches include:

- Character-level contextual feature extraction
- Fixed-window classification
- Sequence labeling models
- Conditional Random Fields (CRF)
- Recurrent Neural Networks (RNN/LSTM)
- Model comparison and optimization

## Dataset

The project uses a German compound segmentation dataset based on the study by Ma et al. (2016).

The dataset contains:

- Training examples with compound words and their segmentation boundaries
- Test examples containing unsegmented German words

Additional unlabeled German text data may be used, subject to the assignment restrictions.

## Project Structure

```text
german-compound-splitting/
│
├── nlp2-a3.train
├── nlp2-a3.test
├── a3-<method>.py
├── a3-<method>.predictions
├── README.md
└── .gitignore
```

## Running the Project

Install the required dependencies:

```bash
pip install numpy pandas scikit-learn
```

Run a segmentation model:

```bash
python a3-<method>.py
```

The script generates a prediction file containing segmented German compound words.

## Applications

German compound splitting is relevant to:

- Natural Language Processing pipelines
- Text preprocessing and tokenization
- Morphological analysis
- Information retrieval
- Machine translation
- Search engine optimization for morphologically complex languages

## Learning Outcomes

This project demonstrates practical experience with:

- Character-level text processing
- Morphological segmentation
- Sequence labeling
- Supervised machine learning
- Context-based feature engineering
- Converting classification outputs into structured text
- German language processing

## Academic Context

Originally developed as part of a university assignment in Statistical Natural Language Processing (SNLP).

The repository has been organized as a portfolio project to demonstrate practical NLP, sequence modeling, and machine learning skills.

---

**Technologies:** Python · NLP · Sequence Labeling · Machine Learning · German Language Processing
