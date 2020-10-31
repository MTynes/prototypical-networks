# Prototypical Networks for Few-shot Learning

This is a modified version of work by [jakesnell](https://github.com/jakesnell/prototypical-networks), which permits the use of a custom dataset.
It has been tested using Google Colab with and without GPU.


The original code implements the NeurIPS paper [Prototypical Networks for Few-shot Learning](http://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf).

As with the original code, citations should be as follows:
```
@inproceedings{snell2017prototypical,
  title={Prototypical Networks for Few-shot Learning},
  author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
 }
 ```
 
 The code expects the directory with all images and the directory containing the label files (train.txt, val.txt, test.txt, or train_val.txt) to be passed to it.
 
 # Usage example, with GPU
 ```
 images_directory = '/content/eeg/all_images'
 lr = 0.0001
 epochs = 10000
 ! python scripts/train/few_shot/run_train_custom_dataset.py --data.data_path {images_directory} --data.labels_path {images_directory} --train.learning_rate {lr} --train.epochs {epochs} --data.cuda 
```

