import torch
from torch.utils.data import DataLoader
from loss import average_dice_coefficient
from dataset import Multimodel_dataset
from res_multipath import csunet
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
best_model = 'best_score.pth'

def inference(model, dataloader,device,eps =1e-6,output_folder='predicted_masks'):
    model.load_state_dict(torch.load(best_model))
    model.eval()
    with torch.no_grad():

        total_dice = 0.0
        total_acc = 0.0
        total_iou = 0.0
        total_specificity = 0.0
        total_sensitivity = 0.0
        total_dice_pre = 0.0
        os.makedirs(output_folder, exist_ok=True)


        progress_bar = tqdm(dataloader)

        for batch in progress_bar:
            img = batch['image'].to(dtype=torch.float32).to(device)
            superpixel_image =  batch['superpixel_image'].to(dtype=torch.float32).to(device)
            mask = batch['mask'].to(dtype=torch.float32).to(device)
            name = batch['image_path'][0]
            #print(name)
            filename = os.path.basename(name)
            assert torch.all((mask == 0) | (mask == 1)), 'Mask contains values other than 0 and 1'

            output_seg = model(img,superpixel_image)

            preds = (torch.sigmoid(output_seg) > 0.5).float()
            preds_np = preds.squeeze().cpu().numpy()
            preds_np = (preds_np * 255).astype('uint8')
            preds_img = np.stack([preds_np] * 3, axis=-1)  # 复制到三个通道
            preds_img_pil = Image.fromarray(preds_img)
            preds_img_pil.save(os.path.join(output_folder, f'{filename}.png'))

            dice_pre = average_dice_coefficient(preds,mask)


            tn = ((preds == 0) & (mask == 0)).sum().item()
            tp = ((preds == 1) & (mask == 1)).sum().item()
            fp = ((preds == 1) & (mask == 0)).sum().item()
            fn = ((preds == 0) & (mask == 1)).sum().item()

            acc = (tp + tn) / (tp + tn + fp + fn + eps)
            iou = tp / (tp + fp+ fn + eps)
            dice = 2*tp / (2*tp + fn + fp + eps)
            se = tp / (tp + fn + eps)
            sp = tn / (tn + fp + eps)





            total_dice_pre += dice_pre
            total_dice += dice
            total_acc += acc
            total_iou += iou
            total_specificity += sp
            total_sensitivity += se



            progress_bar.set_postfix({
                'Dice': dice,
                'Acc': acc,
                'IoU': iou,
                'Spec': sp,
                'Sens': se,
                'dice_p':dice_pre ,
            })
    avg_dicep = total_dice_pre / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    avg_specificity = total_specificity / len(dataloader)
    avg_sensitivity = total_sensitivity / len(dataloader)

    print("Validation dice score:", avg_dice)
    print("Validation accuracy:", avg_acc)
    print("Validation IoU:", avg_iou)
    print("Validation specificity:", avg_specificity)
    print("Validation sensitivity:", avg_sensitivity)
    print("dice_p:", avg_dicep)


    return avg_dice, avg_acc, avg_iou, avg_specificity, avg_sensitivity


if __name__ == '__main__':
    test_img = r'H:\dataset\isic224\new_2017_224_test\img'
    test_super = r'H:\dataset\isic224\new_2017_224_test\super'
    test_cls = r'H:\dataset\isic224\ISIC-2017_Test_cls.csv'
    test_mask = r'H:\dataset\isic224\new_2017_224_test\mask'

    test_dataset = Multimodel_dataset(img_dir=test_img, mask_dir=test_mask, excel_path=test_cls, superpixel_dir=test_super,
                                     transforms=None)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = csunet(num_channels=3, num_class=1, num_cls=3).to(device)

    dice_score = inference(model, test_dataloader,device)



