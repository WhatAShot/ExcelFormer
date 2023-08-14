# ExcelFormer

The pioneering neural network surpassing extremely-tuned XGboost and Catboost on varied tabular datasets.

This repository will include the original implementation and experiment codes of *ExcelFormer*([link](https://arxiv.org/abs/2301.02819))

The implementation of ExcelFormer in the original paper is `bin/excel_former.py`.


## How to test your model

You can test your models by adding them to `bin` directory and `bin/__init__.py`. Keep the same API we used in other models, and write your own evaluation script (`run_default_config_excel.py` as a reference).

## Future work

We will organize our previous works on **tabular prediction** into [Tabular AI Research](https://github.com/pytabular-ai) group for industrial use (e.g. further architecture optimization or acceleration / compilation). If you want to include our model as a baseline in your paper, please use the version in this repository rather than the industrial one in the group repository.


## How to cite ExcelFormer

For now, cite [the Arxiv paper](https://arxiv.org/abs/2301.02819):

```
@article{chen2023excelformer,
  title={ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data},
  author={Chen, Jintai and Yan, Jiahuan and Chen, Danny Ziyi and Wu, Jian},
  journal={arXiv preprint arXiv:2301.02819},
  year={2023}
}
```
