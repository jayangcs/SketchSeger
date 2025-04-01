# Scene Sketch Semantic Segmentation with Hierarchical Transformer

## Dataset

Mosaic-Quickdraw is a scene sketch semantic segmentation dataset that is synthesized using sketch components from [Quick Draw](https://github.com/googlecreativelab/quickdraw-dataset). Mosaic-Quickdraw-Mini is a subset randomly sampled from the Mosaic-Quickdraw dataset. Their specific details are listed below:

**Mosaic-Quickdraw:**

* sample size: 300,000
* categories: [100 different categories](class_list.txt)
* training/validation/testing split: 240,000/30,000/30,000

**Mosaic-Quickdraw-Mini:**

* sample size: 10,000
* categories: [100 different categories](class_list.txt)
* training/validation/testing split: 8,000/1,000/1,000

These datasets can be downloaded from [Hugging Face Dataset Repository](https://huggingface.co/datasets/jayangcs/Mosaic-Quickdraw).

## Code

### Development Environment

Our code is implemented using the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) codebase, and some of the key development environments and package versions are listed below:

* OS: Ubuntu 22.04
* CUDA: 11.3
* Python: 3.9.15
* PyTorch: 1.11.0
* mmengine: 0.3.2
* mmcv: 2.0.0rc3
* mmcls: 1.0.0rc3
* mmsegmentation: 1.0.0rc1

You can use the following commands to install the development environment:

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
mim install mmcv==2.0.0rc3
```

### Checkpoint

The pre-trained checkpoint file for the SketchSeger model can be downloaded from [Hugging Face Model Repository](https://huggingface.co/jayangcs/SketchSeger), and then you need to put it in the `$ROOT_PATH/Checkpoints` directory.

### Benchmark Datasets

* SketchyScene: [Link](https://github.com/SketchyScene/SketchyScene)
* SKY-Scene & TUB-Scene: [Link](https://github.com/drcege/Local-Detail-Perception)

The benchmark datasets need to be placed in the `$ROOT_PATH/Datasets` directory. Note: In order to adapt to the file naming requirements of dataloader, you need to change the names of all files in the `DRAWING_GT` directory of the benchmark datasets from `L0_sample[id].png` to `sample_[id]_drawing.png`. For example, file `L0_sample1.png` needs to be renamed to `sample_1_drawing.png`.

### Training

**Training on single GPU:**

```
python mmsegmentation/tools/train.py [path-to-config-file]
```

Example:

```
python mmsegmentation/tools/train.py mmsegmentation/custom_configs/sketchseger/sketchseger_sketchyscene.py
```

**Training on multiple GPUs:**

```
bash mmsegmentation/tools/dist_train.sh [path-to-config-file] [gpu-num]
```

Example:

```
bash mmsegmentation/tools/dist_train.sh mmsegmentation/custom_configs/sketchseger/sketchseger_sketchyscene.py 4
```

### Testing

```
python mmsegmentation/tools/test.py [path-to-config-file] --checkpoint [path-to-checkpoint] --work-dir [path-to-work-dir]
```

Example:

```
python mmsegmentation/tools/test.py mmsegmentation/custom_configs/sketchseger/sketchseger_sketchyscene.py --checkpoint Checkpoints/sketchseger_sketchyscene_latest.pth --work-dir outputs/sketchseger_sketchyscene_test
```

### Inference

```
python mmsegmentation/custom_tools/custom_inference.py
```

## Citation

If you find our dataset or code useful for your research, please consider citing this paper.

```
@article{yang2023scene,
  title={Scene sketch semantic segmentation with hierarchical Transformer},
  author={Yang, Jie and Ke, Aihua and Yu, Yaoxiang and Cai, Bo},
  journal={Knowledge-Based Systems},
  volume={280},
  pages={110962},
  year={2023},
  publisher={Elsevier}
}
```
