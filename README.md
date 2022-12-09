# Overview
The implementation of [Integrated Gradient](https://arxiv.org/abs/1703.01365)

# Example
![](assets/n01669191_46.JPEG)
![](assets/heatmap.jpg)

# Usage
- The example image is generate from mobilenetv2:  
```
python show.py -d cpu -img assets/n01669191_46.JPEG -step 20
```

- for custom model  
model path is a file including weight and architecture. 
```
python show.py -d cpu -img assets/n01669191_46.JPEG -step 20 -m {your model path}
```
- Get predict label  
  My program will print the predict label, you can refer to `show.py`.  
  For imagenet label to class name, you can refer to [class id to name](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).
  
# Note
- Remenber to check whether the image preprocess is the same as yours, if not, you should alert the preprocess in the `show.py` or the result will be wrong.
- If you have cuda, you can just replace the "cpu" to "cuda".
- The bigger the `step` is, the more memory will be used, thus, don't set `step` too high.  
According to the paper, 20 steps are enough to show a good heatmap, so I set `step` to 20 by default.

# All inplementation of CAM method
| Method   | Framework |
| :-----:  | :--------: |
| Grad-CAM | [Pytorch](https://github.com/shyhyawJou/GradCAM-pytorch)/[Tensorflow](https://github.com/shyhyawJou/GradCAM-Tensorflow) | 
| Eigen-CAM | [Pytorch](https://github.com/shyhyawJou/EigenCAM-Pytorch)/[Tensorflow](https://github.com/shyhyawJou/Eigen-CAM-Tensorflow) |
| Ablation-CAM | [Tensorflow](https://github.com/shyhyawJou/Ablation-CAM-Tensorflow) |
| Integrated Gradient | [Pytorch](https://github.com/shyhyawJou/Integrated-Gradient-Pytorch) |
