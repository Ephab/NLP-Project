# Multilabel Emotion Classification (NLP Project, Section 7MA1, Group 3)

Predicts which emotions an English comment expresses, as one or more of seven groups
(anger, disgust, fear, joy, sadness, surprise, neutral), using the GoEmotions dataset.
We compare TF-IDF with logistic regression, an RNN with pretrained GPT-2 embeddings, and fine-tuned RoBERTa.

Team: Mohammed Firas Sawab (leader), Rashed Ali Alkhathlan, Qasim Mustafa Al-Mahfoodh.

## Files

| File | What it is |
|---|---|
| `notebooks/Emotions_classification.ipynb` | Main notebook: data, all methods, results, error analysis, Try it |
| `notebooks/Emotions_inference.ipynb` | Try the trained models on new text without training |
| `requirements.txt` | Package versions used for the reported results |
| Release [`models-v1`](https://github.com/Ephab/NLP-Project/releases/tag/models-v1) | Trained models (`saved_models.zip`, about 460 MB) |

## Try the models (about 2 minutes, no GPU needed)

Open `notebooks/Emotions_inference.ipynb` in Google Colab
([link](https://colab.research.google.com/github/Ephab/NLP-Project/blob/main/notebooks/Emotions_inference.ipynb))
and choose Runtime → Run all. It downloads the trained models from the release into a relative
`saved_models/` folder, then the Try it cell accepts any sentence. Empty input prints a helpful message.

## Reproduce the results (about 45 minutes)

1. Open `notebooks/Emotions_classification.ipynb` in Google Colab.
2. Runtime → Change runtime type → T4 GPU.
3. Runtime → Run all.

The dataset (GoEmotions, Kaggle mirror `debarshichanda/goemotions`, version 6) is downloaded by the notebook with
`kagglehub`. Seeds are fixed to 42. To skip training, run Section 1 (Setup) and then Section 8 (Try it), which
loads the released models instead.

## Sources

- D. Demszky et al., "GoEmotions: A Dataset of Fine-Grained Emotions," ACL 2020.
- Y. Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach," 2019 (weights: `roberta-base`).
- A. Radford et al., "Language Models are Unsupervised Multitask Learners," 2019 (GPT-2 tokenizer and embeddings).
- Libraries: scikit-learn, PyTorch, Hugging Face Transformers, pandas, matplotlib.
