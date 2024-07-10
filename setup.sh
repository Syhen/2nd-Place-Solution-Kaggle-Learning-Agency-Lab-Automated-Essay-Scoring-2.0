#!/bin/bash

if pip list | grep kaggle; then
    echo "kaggle is already installed."
else
    echo "Installing kaggle..."
    pip install kaggle
fi

pip install -r ./requirements.txt

mkdir my_datasets
cd my_datasets
kaggle competitions download -c learning-agency-lab-automated-essay-scoring-2
unzip -o learning-agency-lab-automated-essay-scoring-2.zip -d "./learning-agency-lab-automated-essay-scoring-2"
rm learning-agency-lab-automated-essay-scoring-2.zip

kaggle datasets download -d syhens/aes2-external
unzip -o aes2-external.zip -d "./external"
rm aes2-external.zip

cd ..
python make_folds.py

python -m nltk.downloader punkt
