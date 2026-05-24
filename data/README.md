# Dataset Files

This directory contains dataset adapters and preprocessing scripts. We may not
have permission to redistribute all dataset files in the anonymous release,
so some datasets must be downloaded by the user before preprocessing.

The following three datasets are already included in this release:

- `adult`
- `california`
- `diamonds`

For the remaining datasets:

- `okcupid_stem`: a download script is provided in `data/okcupid_stem/`.
- `diabetes_130US`: download the raw dataset from the UCI repository:
  https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008.
  We construct the processed dataset ourselves following the publisher's paper;
  the preprocessing scripts are included in `data/diabetes_130US/`.
- `home_credit`: download the data from the Kaggle competition:
  https://www.kaggle.com/competitions/home-credit-default-risk.

