from PIL import Image
import argparse

import torch
from torchvision import transforms as T
from torchvision.models import mobilenet_v2

from ig import Integrated_Gradient



def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', help='pytorch model path (a file incusive of weight and architecture)')
    parser.add_argument('-d', default='cpu', help='device ("cpu" or "cuda")')
    parser.add_argument('-img', help='img path')
    parser.add_argument('-step', default=20, type=int, help='number of steps in the path integral')
    return parser.parse_args()
    


def main():
    arg = get_arg()
    
    preprocess = T.Compose([T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], 
                                        [0.229, 0.224, 0.225])])
    
    device = arg.d
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('There is no cuda !!!')
    
    if arg.m is None:
        model = mobilenet_v2(True).eval()
    else:
        model = torch.load(arg.m).eval()
    
    ig = Integrated_Gradient(model, arg.d, preprocess, arg.step)
    
    print('\ndevice:', arg.d)
    print('img:', arg.img)
    
    img = Image.open(arg.img).convert('RGB')
    baseline = torch.zeros((1 , 3, 224, 224), requires_grad=True)
    
    # output is torch Tensor, heatmap is ndarray
    output, heatmap = ig.get_heatmap(img, baseline)
    print('\nPredict label:', output.max(1)[1].item())

    w, h = img.size
    heatmap = Image.fromarray(heatmap)
    heatmap.save('heatmap.jpg')
    result = Image.new('RGB', (2 * w, h))
    result.paste(img)
    result.paste(heatmap, (w, 0))
    result.show()
    

if __name__ == "__main__":
    main()
