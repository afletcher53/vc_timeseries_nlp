# Quickstart

1. Run `refactor_vetcompass_to_project_form.ipynb`
2. Run `create_data_windows.ipynb`
3. Run `exploratory_data_analysis.ipynb`
4. Run `single_timestep_classifiers.ipynb`
5. Run `single_timestep_bert_classifier.ipynb`
6. Run `multi_timestep_classifiers.ipynb`

# Methodology

## Refactor Vetcompass to Project Form
### `refactor_vetcompass_to_project_form.ipynb`

Refactors vetcompass data to format for the project

## Create Data Windows
### `create_data_windows.ipynb`

Data is sections where every visit creates an input, with n timesteps before it.
Classification is determined by text within the n timesteps output window.
Data is vectorised using pretrained Glove 300 dimension
Test?Train/Validation set produced and saved in data

## Exploratory Data Analysis
### `exploratory_data_analysis.ipynb`

Produces data based on copora - X_train dataset.
Visit culmative day of year graph
Top N words graph overall corpora, also by category (revisit / no revisit)
Shannon Diversity Equation demonstration for data imbalance, showing that SMOTE improves SDE scoring.

## Single Timestep Classifiers
### `single_timestep_classifiers.ipynb`

Run a Single LSTM, BiLSTM, LSTM (stacked), BiLSTM (stacked) and CNN on the dataset.

## Single Timestep Bert Classifier
### `single_timestep_bert_classifier.ipynb`
Run BERT on the dataset.

## Single Timestep Classifiers
### `single_timestep_bert_classifier.ipynb`

Run a Timedistributed LSTM, BiLSTM, LSTM (stacked), BiLSTM (stacked) and CNN on the dataset.


# In progress / Todo

### File: `run_models.ipynb`

Save produced models
Produce training graphs for each model
Implement model evaluation and selection of highest scoring model

### File: `pretrain_glove_on_training_data.ipynb`

Uses Mittens to pretrain Glove Embeddings on the training corpora.

- Sort out GLOVE file saving


### File: `saliency_measurement.ipynb`

Need to create this file and implement saliency metrics
