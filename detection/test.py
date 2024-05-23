
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.nn import functional as F
import torch.nn as nn
# from knn_cuda import KNN
import pickle
from sklearn import metrics
import yaml
from dataloader import FaceDataset
from torch.utils.data import DataLoader
from model_zoo import (SelfBlendedModel, 
                       MAT, ICT, 
                       RosslerModel, 
                       ForgeryNet, 
                       CADDM, 
                       resnet50,
                       calculate_roc_ex, 
                       evaluate_new)
from utils import (IsotropicResizeTorch, PadIfNeeded,ToIntImage)
from sklearn.metrics import roc_curve, auc
from loops import evaluation, ict_evaluate

import sys
sys.path.append("./Capsule-Forensics-v2/")
import model_big as CapSule
from torch.utils.model_zoo import load_url
from MCXAPI.models import API_Net
from LGrad.CNNDetection.networks import resnet as LGrad
from icpr2020dfdc.architectures import fornet, weights

def parse_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process to test a model with a folder of images')
    VALID_MODELS = ['selfblended', 'mat', 'ict', 'rossler', 'forgerynet', 'capsule', 'caddm', 'ccvit', 'add']
    parser.add_argument('--model-name', default=None, choices=VALID_MODELS, type=str, help='the model name to test')
    parser.add_argument('--test-folder',  type=str, default=None, metavar='S',  nargs='+', help='The folder of images to test the model')
    parser.add_argument('--batch-size', default=32,  type=int, help='batch size to test the model')
    parser.add_argument('--penul-ft', action='store_true', help='Return penultimate ft for plotting')
    parser.add_argument('--sampling-rate', default=1,  type=int, help='sampling frequecy each video')
    parser.add_argument('--data-type', default='created',  type=str,choices=['created', 'collected', 'cdfv2', 'dfdc'],
                         help='dataset to test')
    args = parser.parse_args()
    return args




def main(args):
    print("Test model: ", args.model_name.upper())
    softmax = True
    if args.model_name == 'selfblended':
        model = SelfBlendedModel().cuda()

        img_size = 380
        transformer = transforms.Compose([
                            transforms.Resize((img_size, img_size)),
                            transforms.ToTensor(),
                            ])
        fake_class = 1
        model.eval()
        use_bgr = False
    elif args.model_name == 'rossler':
        model = RosslerModel(modelchoice='xception').cuda()
        

        img_size = 299
        transformer = transforms.Compose([
                            transforms.Resize((img_size, img_size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5] * 3, [0.5] * 3)
                            ])
        fake_class = 1
        model.eval()
        use_bgr = False
    elif args.model_name == 'mat':
        model = MAT().cuda()
        
        img_size = 380
        transformer = transforms.Compose([
                            transforms.Resize((img_size, img_size)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
                            ])
        fake_class = 1
        model.eval()
        use_bgr = False
        
    elif args.model_name == 'ccvit':
        sys.path.append("Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/cross-efficient-vit")
        from cross_efficient_vit import CrossEfficientViT
        with open("Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/cross-efficient-vit/configs/architecture.yaml", 'r') as ymlfile:
            config = yaml.safe_load(ymlfile)
        model = CrossEfficientViT(config=config)
        model.load_state_dict(
            torch.load("pretrained-weight/ccvit/cross_efficient_vit.pth"),
            strict=True)
        model.eval()
        model = model.cuda()
        
        img_size = 224
        transformer = transforms.Compose([
                            IsotropicResizeTorch(img_size),
                            PadIfNeeded(img_size, img_size, fill=(0, 0, 0)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.,0.,0.],std=[1.0,1.0,1.0]),
                            ])
        fake_class = 1
        softmax = False
        model.eval()
        use_bgr = True
        
    elif args.model_name == 'ict':
        model = ICT().cuda()
        
        img_size = 112
        transformer = transforms.Compose([
                            transforms.Resize((img_size, img_size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        ])
        fake_class = 1
        model.eval()
        use_bgr = False
        
    elif args.model_name == 'caddm':
        model = CADDM(2, backbone='resnet34').cuda()
        pretrained_model = 'pretrained-weight/iil/resnet34.pkl'
        checkpoint = torch.load(pretrained_model)
        model.load_state_dict(checkpoint['network'])
        img_size = 224
        transformer = transforms.Compose([
                            transforms.Resize((img_size, img_size)),
                            transforms.ToTensor()
                        ])
        fake_class = 1
        model.eval()
        use_bgr = True
    elif args.model_name == 'add':
        from torchvision.models import resnet50
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = model.cuda()
        pretrained_model = 'pretrained-weight/add/deepfakes_c23resnet50_kd_valacc_img128_kd21_freq_swd_best.pth'
        checkpoint = torch.load(pretrained_model)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        img_size = 128
        transformer = transforms.Compose([
                            transforms.Resize((img_size, img_size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        ])
        fake_class = 1
        model.eval()
        use_bgr = False
        
    elif args.model_name == 'capsule':
        vgg_ext = CapSule.VggExtractor().cuda()
        capnet = CapSule.CapsuleNet(2, 0).cuda()
        capnet.load_state_dict(torch.load(os.path.join('Capsule-Forensics-v2/checkpoints/binary_faceforensicspp','capsule_21.pt')))
        capnet.eval()
        model = [vgg_ext, capnet]
        img_size = 300

        transformer =  transforms.Compose([
                        transforms.Resize((img_size, img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ])
        fake_class = 1
        use_bgr = False

    elif  args.model_name =='forgerynet':
        model = ForgeryNet(num_classes=2).cuda()

        weight_path = "pretrained-weight/forgerynet/ckpt_iter.pth.tar"
        weight = torch.load(weight_path)['state_dict']
        updated_weight = dict()
        for k in weight.keys(): updated_weight[k.replace('module.', '')] = weight[k]
        model.load_state_dict(updated_weight, strict=True)
        img_size = 299
        transformer = transforms.Compose([
                            transforms.Resize((img_size, img_size)),
                            transforms.ToTensor(),
                            # transforms.Normalize([0.5] * 3, [0.5] * 3)
                            ])
        fake_class = 1
        model.eval()
        use_bgr = False

 
    elif args.model_name == 'mcx':
        from model_zoo.mcx import API_Net
        model = API_Net(num_classes=5, model_name='xception').cuda()
        model.conv = nn.DataParallel(model.conv)
        pretrained_model = 'pretrained-weight/mcx/model_mcx-api-rgb.tar'
        checkpoint = torch.load(pretrained_model)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        img_size = 448
        transformer = transforms.Compose([
                    transforms.Resize([512, 512]),
                    transforms.RandomCrop([img_size, img_size]),
                    # transforms.RandomHorizontalFlip(), not common to use random in test
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)
                    )])
        fake_class = 0 # use the real class in inverse in the loop
        model.eval()
        use_bgr = False
    
    elif 'lgrad' in model_name :
        
        model = LGrad.resnet50(num_classes=1)
        model.load_state_dict(torch.load(args.resume, map_location='cpu'), strict=True) # args.resume = 'LGrad-1class-Trainon-Progan_horse.pth'
        model.cuda()
        transform = transforms.Compose([
                    transforms.Resize([256, 256]),
                    transforms.RandomCrop([224, 224]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                    

    elif args.model_name == 'effb4att':
        net_model = 'EfficientNetAutoAttB4ST'
        model_path = weights.weight_url['{:s}_{:s}'.format(net_model, 'FFPP')]
        checkpoint = load_url(model_path, map_location='cuda', check_hash=True)
        model = getattr(fornet, net_model)().cuda()
        model.load_state_dict(checkpoint['net'], strict=True)
        img_size = 224
        transformer =transformer =  transforms.Compose([
                        transforms.Resize((img_size, img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ])
        fake_class = 1
        softmax = False
        model.eval()
        use_bgr = False

    valid_dataloader = DataLoader(FaceDataset(args.test_folder, transform=transformer, use_bgr=use_bgr, sampling_rate=args.sampling_rate),
                            batch_size=args.batch_size, shuffle=False)
    if not args.penul_ft:
        if args.model_name != 'ict':
            ACC, ACC_best, AUC, = evaluation(model, valid_dataloader, fake_class, args.model_name, softmax=softmax)
        else:
            ACC, ACC_best, AUC,  = ict_evaluate(model, valid_dataloader)
        return (ACC, ACC_best, AUC)
    

    else:
        if args.model_name != 'ict':
            ACC, ACC_best, AUC, ft_list, lb_list = evaluation(model, valid_dataloader, fake_class, args.model_name, softmax=softmax, penul_ft=True )
        else:
            ACC, ACC_best, AUC, ft_list, lb_list = ict_evaluate(model, valid_dataloader)
        return (ACC, ACC_best, AUC, ft_list, lb_list)
    


    
def test_folders(args):
    """
    Test all the generated data and export them into csv file with 4 columns
    - Dataset
    - ACC
    - ACC @best
    - AUC
    """
    print("TEST UPON SETTING")
    if args.penul_ft:
        print("Save penultimate features")
        penul_data = dict()
        assert args.method in ['mat'], "Not implemented this method for intermediate return features ... "
    out_results = pd.DataFrame({
        "Dataset":[], "Acc": [], "Acc_best": [], "AUC": []
    })
    for data_name in ("DeepFaceLab", "Dfaker", "Faceswap", "FOM_Animation", "FOM_Faceswap", "FSGAN", "LightWeight"):
        args.test_folder = ["/datasets/Stabilized/real/",
                    f"datasets/Stabilized/{data_name}/"]
        if "lgrad" in args.model_name:
            lgrad_path = "/datasets/LGrad/Stabilized/" # To test Lgrad on Stabilized dataset, pre-processing our dataset and saving in this directory is needed.
            args.test_folder = [lgrad_path +f"real/", 
                                    lgrad_path +f"{data_name}/"]
        print(args.test_folder)
        if not args.penul_ft:
            ACC, ACC_best, AUC  = main(args)

        else:
            penul_data[data_name] = dict()
            ACC, ACC_best, AUC, ft_list, lb_list = main(args)
            penul_data[data_name]['ft'] = np.asarray(ft_list)
            penul_data[data_name]['lb'] = np.asarray(lb_list)

        out_results = out_results.append({"Dataset":data_name, 
                                            "Acc": np.round(ACC,2), 
                                            "Acc_best": np.round(ACC_best,2), 
                                            "AUC": np.round(AUC,2)}, ignore_index=True)
    out_results = out_results.append({"Dataset":"Avg", 
                                        "Acc": f"{out_results.Acc.mean():.2f} ({out_results.Acc.std():.2f})", 
                                        "Acc_best": f"{out_results.Acc_best.mean():.2f} ({out_results.Acc_best.std():.2f})", 
                                        "AUC": f"{out_results.AUC.mean():.2f} ({out_results.AUC.std():.2f})"}, 
                                        ignore_index=True)
    out_results.to_csv(f"predictions/{args.model_name}.csv", index=False)
    if args.penul_ft:
        with open(f"predictions/{args.model_name}_penultimate_ft.pkl", 'wb') as file:
            pickle.dump(penul_data, file)

def itw_test_folders(args):
    """
    Test in the wild dataset
    - Dataset : ITW
    - ACC
    - ACC @best
    - AUC
    """
    print("TEST UPON SETTING")
    out_results = pd.DataFrame({
        "Dataset":[], "Acc": [], "Acc_best": [], "AUC": []
    })

    args.test_folder = ["/datasets/RDFW23/real/",
                f"/datasets/RDFW23/fake/"]
    print(args.test_folder)
    ACC, ACC_best, AUC = main(args)
    out_results = out_results.append({"Dataset":"itw", 
                                        "Acc": np.round(ACC,2), 
                                        "Acc_best": np.round(ACC_best,2), 
                                        "AUC": np.round(AUC,2)}, ignore_index=True)

    out_results.to_csv(f"predictions/{args.model_name}(in the wild).csv", index=False)

def dfdcp_test_folders(args):
    """
    Test in the wild dataset
    - Dataset : DFDCP
    - ACC
    - ACC @best
    - AUC
    """
    print("TEST UPON SETTING")
    out_results = pd.DataFrame({
        "Dataset":[], "Acc": [], "Acc_best": [], "AUC": []
    })

    args.test_folder = ["/datsets/dfdc/real/",
                f"dataset/dfdc/fake/"]
    if 'lgrad' in args.model_name:
        lgrad_path = "/datasets/LGrad/dfdcp/" 
        args.test_folder = [f"{lgrad_path}real/",
                            f"{lgrad_path}fake/"]
    print(args.test_folder)
    ACC, ACC_best, AUC = main(args)
    out_results = out_results.append({"Dataset":"dfdcp", 
                                        "Acc": np.round(ACC,2), 
                                        "Acc_best": np.round(ACC_best,2), 
                                        "AUC": np.round(AUC,2)}, ignore_index=True)

    out_results.to_csv(f"predictions/{args.model_name}(dfdcp).csv", index=False)
    
def cdf_test_folders(args):
    """
    Test in the wild dataset
    - Dataset : CelebDfv2
    - ACC
    - ACC @best
    - AUC
    """
    print("TEST UPON SETTING")
    out_results = pd.DataFrame({
        "Dataset":[], "Acc": [], "Acc_best": [], "AUC": []
    })

    args.test_folder = ["datasets/CelebDF-v2/real/",
                f"datasets/CelebDF-v2/fake/"]
    if 'lgrad' in args.model_name:
        lgrad_path = "/datasets/LGrad/CelebDF-v2/" 
        args.test_folder = [f"{lgrad_path}real/",
                            f"{lgrad_path}fake/"]
    
    print(args.test_folder)
    ACC, ACC_best, AUC = main(args)
    out_results = out_results.append({"Dataset":"dfdcp", 
                                        "Acc": np.round(ACC,2), 
                                        "Acc_best": np.round(ACC_best,2), 
                                        "AUC": np.round(AUC,2)}, ignore_index=True)

    out_results.to_csv(f"predictions/{args.model_name}(CDFv2).csv", index=False)
    

if __name__ == '__main__':
    args = parse_args()
    if args.data_type =='created': # Test the created dataset
        if args.test_folder is not None:
            main(args)
        else:
            test_folders(args) 
    elif args.data_type =='dfdc': # Test the updated dataset
        dfdcp_test_folders(args)
    elif args.data_type =='cdfv2': # Test the updated dataset
        cdf_test_folders(args)
    elif 'collected': # Test the collected dataset
        itw_test_folders(args)
