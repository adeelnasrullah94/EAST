# Composite Music Symbol Localization using EAST

### Contents
1. [Installation](#installation)
2. [Download](#download)
3. [Train](#test)
4. [Test](#train)

### Installation
1. Any version of tensorflow version > 1.0 should be ok.

### Dataset
DeepScores Dense Extended dataset needs to be preprocessed prior to training EAST for music symbol localization. In order to run the preprocessing script run
``` pyhton preprocess.py```

### Download
1. Resnet V1 50 provided by tensorflow slim: [slim resnet v1 50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)

### Train
In order to train the model, you should provide the dataset path, in the dataset path, a separate gt text file should be provided for each image
and run

```
python multigpu_train.py --gpu_list=0 --input_size=512 --batch_size_per_gpu=14 --checkpoint_path=/tmp/east_deepscores_resnet_v1_50_rbox/ \
--text_scale=512 --training_data_path=/data/ocr/deepscores_localization/ --geometry=RBOX --learning_rate=0.0001 --num_readers=24 \
--pretrained_model_path=/tmp/resnet_v1_50.ckpt
```

If you have more than one gpu, you can pass gpu ids to gpu_list(like --gpu_list=0,1,2,3)
(Note: Resnet V1 50 pretrained model was used in the implementation of this paper).

### Test
run
```
python eval.py --test_data_path=/tmp/images/ --gpu_list=0 --checkpoint_path=/tmp/east_deepscores_resnet_v1_50_rbox/ \
--output_dir=/tmp/
```

a text file will be then written to the output path.

### Evaluation
In order to calculate F1-Score, the output text files obtained from above script as well as gt files for test images are required. To obtain precision, recall and F1-Score run
``` pyhton score.py ```