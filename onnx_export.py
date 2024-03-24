import os
import sys
import argparse

import torch.onnx

from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite

from vision.ssd.config import mobilenetv1_ssd_config


# parse command line
parser = argparse.ArgumentParser()
parser.add_argument('--net', default="mb1-ssd",
                    help="The network architecture, it can be mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or mb3-ssd-lite.")
parser.add_argument('--input', type=str, default='',
                    help="path to input PyTorch model (.pth checkpoint)")
parser.add_argument('--output', type=str, default='',
                    help="desired path of converted ONNX model (default: <NET>.onnx)")
parser.add_argument('--labels', type=str, default='labels.txt',
                    help="name of the class labels file")
parser.add_argument('--batch', type=int, default=1,
                    help="batch size of the model to be exported (default=1)")
parser.add_argument('--model-dir', '--model', type=str, default='',
                    help="directory to look for the input PyTorch model in, and export the converted ONNX model to (if --output doesn't specify a directory)")

args = parser.parse_args()
print(args)

# set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"=> running on device {device}")

# format input model paths
if args.model_dir:
    args.model_dir = os.path.expanduser(args.model_dir)

    # find the checkpoint with the lowest loss
    if not args.input:
        best_loss = 10000
        for index, file in enumerate(os.listdir(args.model_dir)):
            if not file.endswith(".pth"):
                continue
            try:
                loss = float(file[file.rfind("-")+1:len(file)-4])
                if loss < best_loss:
                    best_loss = loss
                    args.input = os.path.join(args.model_dir, file)
            except ValueError:
                args.input = os.path.join(args.model_dir, file)
                continue

        if not args.input:
            raise IOError(
                f"couldn't find valid .pth checkpoint under '{args.model_dir}'")

        print(f"=> found best checkpoint with loss {best_loss} ({args.input})")

    # append the model dir (if needed)
    if not os.path.isfile(args.input):
        args.input = os.path.join(args.model_dir, args.input)

    if not os.path.isfile(args.labels):
        args.labels = os.path.join(args.model_dir, args.labels)

# determine the number of classes
class_names = [name.strip() for name in open(args.labels).readlines()]
num_classes = len(class_names)

# construct the network architecture
print(f"=> creating network:  {args.net}")
print(f"=> num classes:       {num_classes}")

if args.net == 'mb1-ssd':
    mobilenetv1_ssd_config.set_image_size(300)
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif args.net == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif args.net == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif args.net == 'mb3-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)

# load the model checkpoint
print(f"=> loading checkpoint:  {args.input}")

net.load(args.input)
net.to(device)
net.eval()

# create example image data
dummy_input = torch.randn(args.batch, 3, 300, 300).cuda()

# format output model path
if not args.output:
    args.output = args.net + '.onnx'

if args.model_dir and args.output.find('/') == -1 and args.output.find('\\') == -1:
    args.output = os.path.join(args.model_dir, args.output)

# export to ONNX
input_names = ['input_0']
output_names = ['scores', 'boxes']

print("=> exporting model to ONNX...")
torch.onnx.export(net, dummy_input, args.output, verbose=True,
                  input_names=input_names, output_names=output_names)
print(f"model exported to:  {args.output}")
