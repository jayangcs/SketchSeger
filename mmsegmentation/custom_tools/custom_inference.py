import os
import numpy as np
import cv2
from PIL import Image
from custommmseg.apis import init_model, inference_model, show_result_pyplot
from mmseg.utils import register_all_modules
import scipy.io as io


def visualize_semantic_segmentation(pred_label, palette, save_path, black_bg=False):
    """
    tool for visualizing semantic segmentation for a given label array

    :param pred_label: [H, W], contains [0-nClasses], 0 for background
    :param palette: the palette from dataset
    :param black_bg: the background is black if set True
    :param save_path: path for saving the image
    """
    visual_image = np.zeros((pred_label.shape[0], pred_label.shape[1], 3), dtype=np.uint8)
    if not black_bg:
        visual_image.fill(255)

    # assign color to drawing regions
    visual_image[pred_label != 255] = palette[pred_label[pred_label != 255]]

    # save visualization
    visual_image = Image.fromarray(visual_image, 'RGB')
    visual_image.save(save_path)


def inference_sketch(imgs, save_path="../outputs/inference_output"):
    # init model
    model = init_model(config_path, checkpoint_path, device="cuda:0")

    if imgs[0].split(".")[-1] == 'mat':
        save_path_label = os.path.join(save_path, 'label')
        os.makedirs(save_path_label, exist_ok=True)
        for label_path in imgs:
            if label_path.split(".")[-1] == 'mat':
                label = io.loadmat(label_path)['CLASS_GT']
                label[label == 0] = 255
                label -= 1
                label[label == 254] = 255
                save_path_label_temp = os.path.join(save_path_label, label_path.split("/")[-1].split(".")[-2] + '.png')
                visualize_semantic_segmentation(label, np.array(model.dataset_meta['palette']), save_path_label_temp)
    else:
        # make directory
        save_path_foreground = os.path.join(save_path, 'foreground')
        save_path_full = os.path.join(save_path, 'full')
        os.makedirs(save_path_foreground, exist_ok=True)
        os.makedirs(save_path_full, exist_ok=True)

        # inference
        results = inference_model(model, imgs)

        for result in results:
            pred_label = result.get('pred_sem_seg').data.cpu().numpy().astype(np.uint8).squeeze()
            img = result.get('img_path')
            img_np = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            if len(img_np.shape) == 3:
                pred_label[img_np[:, :, 0] != 0] = 255
            else:
                pred_label[img_np[:, :] != 0] = 255
            save_path_foreground_temp = os.path.join(save_path_foreground, img.split("/")[-1])
            visualize_semantic_segmentation(pred_label, np.array(model.dataset_meta['palette']), save_path_foreground_temp)
            save_path_full_temp = os.path.join(save_path_full, img.split("/")[-1])
            show_result_pyplot(model, img, result, show=False, out_file=save_path_full_temp)


def inference_sketch_dir(dir_path, save_path="../outputs/inference_output", max_infer_num=None):
    files = os.listdir(dir_path)

    # set the number of images to infer
    if max_infer_num:
        files = files[:max_infer_num]

    # add dir_path to file name
    img_path_list = [os.path.join(dir_path, file) for file in files]
    inference_sketch(img_path_list, save_path)


if __name__ == '__main__':
    register_all_modules(init_default_scope=True)

    config_path = "path_to_config_file"
    checkpoint_path = "path_to_checkpoint_file"

    inference_sketch_dir("input_path", save_path="output_path", max_infer_num=50)