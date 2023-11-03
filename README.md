# ExcelFormer

This repository will include the original implementation and experiment codes of [**ExcelFormer**](https://arxiv.org/abs/2301.02819). TabFormer is a pioneering neural network can surpass extensively-tuned XGboost, Catboost, and most tuned previous deep learning approaches on most of tabular data prediction tasks, in the supervised learning manner. It can be a go-to choice on tabualr dataset prediction competitions (e.g., Kaggle).

Even without hyper-parameter tuning, TabFormer performs comparable to tuned models. After hyper-parameter tuning, TabFormer typically outperforms them.

The implementation of TabFormer in the original paper is `bin/excel_former.py`.


## How to test your model

You can test your models by adding them to `bin` directory and `bin/__init__.py`. Keep the same API we used in other models, and write your own evaluation script (`run_default_config_excel.py` as a reference).

## Future work

We will organize our previous works on **tabular prediction** into [Tabular AI Research](https://github.com/pytabular-ai) group for industrial use (e.g. further architecture optimization or acceleration / compilation). If you want to include our model as a baseline in your paper, please use the version in this repository rather than the industrial one in the group repository.
