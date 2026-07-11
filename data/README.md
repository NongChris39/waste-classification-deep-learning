# Dataset directory

The dataset is not stored in this Git repository.

Run the dataset download and split steps from the main notebook to generate the following local folders:

```text
data/
├── raw/
├── train/
├── validation/
└── test/
```

These folders are excluded from Git because they contain generated or downloaded image data.

The dataset is downloaded with `kagglehub` using the identifier configured in `src/download_dataset.py`.
