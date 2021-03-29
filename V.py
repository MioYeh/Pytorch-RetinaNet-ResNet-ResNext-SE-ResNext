import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
import prettytable as pt
import xml.etree.ElementTree as ET

path_xml = '/home/mio861225tw/VGH_DATASET/old_data_distribution/old_T_V_T/Test_Labels/'
output_image = '/home/mio861225tw/pytorch-retinanet/0312_SE_TEST/'
# '/home/mio861225tw/VGH_DATASET/Prediction_image/'
# Valid_predict/' #Prediction_image/'
fontSize = 1
fontBold = 2
labelColor = (0,0,255)
write_path = []
TP = 0
FP = 0
array_TP = 0
array_FP = 0
total = 0
def load_classes(csv_reader):
    result = {}  

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
#     cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
#     cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def detect_image(image_path, model_path, class_list):
    global TP
    global FP
    global array_TP
    global array_FP
    global total
    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    model = torch.load(model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()
    n = 0
    for img_name in os.listdir(image_path):

        image = cv2.imread(os.path.join(image_path, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tree = ET.parse(path_xml + img_name.split('.')[0] +'.xml')
        if image is None:
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
#         min_side = 608
#         max_side = 1024
        min_side = 512 
        max_side = 1920
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        root = tree.getroot()
        xmin = []
        xmax = []
        ymin = []
        ymax = []
        origin = []
        for elem in tree.iter(tag='xmin'):
            xmin.append(elem.text)
        for elem in tree.iter(tag='xmax'):
            xmax.append(elem.text)
        for elem in tree.iter(tag='ymin'):
            ymin.append(elem.text)
        for elem in tree.iter(tag='ymax'):
            ymax.append(elem.text)
        
        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()
#             print(image.shape, image_orig.shape, scale)
            scores, classification, transformed_anchors = model(image.cuda().float())
#             print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)
            total += len(xmin)
            for j in range(idxs[0].shape[0]):
                score_iou = []
                for x in range(len(xmin)) :
                    cv2.rectangle(image_orig, (int(xmin[x]), int(ymin[x])), (int(xmax[x]), int(ymax[x])), (0, 255, 0), 2)
                    bbox = transformed_anchors[idxs[0][j], :]

                    x1 = int(bbox[0] / scale)
                    y1 = int(bbox[1] / scale)
                    x2 = int(bbox[2] / scale)
                    y2 = int(bbox[3] / scale)

                    x_1 = max(int(xmin[x]), x1)
                    y_1 = max(int(ymin[x]), y1)
                    x_2 = min(int(xmax[x]), x2)
                    y_2 = min(int(ymax[x]), y2)

                    x21 = x_2 - x_1
                    y21 = y_2 - y_1
                    w = max(0, x_2 - x_1)
                    h = max(0, y_2 - y_1)
                    interArea = w * h
                    boxAArea = (int(xmax[x]) - int(xmin[x]) + 1) * (int(ymax[x]) - int(ymin[x]) + 1)
                    boxBArea = (x2- x1 + 1) * (y2- y1 + 1)  
                    iou = interArea / float(boxAArea + boxBArea - interArea)
                    score_iou.append(iou)
                    if iou >= 0.5:
                        TP += 1
                        cv2.putText(image_orig, str('%.2f'%iou), (x1, y1-5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                    elif iou < 0.5:
                        FP += 1


                    label_name = labels[int(classification[idxs[0][j]])]
                    score = scores[j]
                    caption = '{} {:.3f}'.format(label_name, score)
                    draw_caption(image_orig, (x1, y1, x2, y2), caption)
                    cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

                if max(score_iou) >= 0.5:
                    array_TP += 1
                else:
                    array_FP += 1 

#             print(array_TP, array_FP )

            image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)        
            cv2.imwrite(output_image + img_name, image_orig) 
            n += 1
    TP = array_TP
    FP = array_FP
    FN = total-array_TP
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1_score = 2 * Precision * Recall / (Precision + Recall)
    
    tb1 = pt.PrettyTable()
    tb1.field_names = ["Total Label", "True Positive", "False Positive", "True Negative", "Precision", "Recall", "F1-score"]
    tb1.add_row([total, TP, FP , FN, '%.2f' % Precision, '%.2f' % Recall, '%.2f' % F1_score])
    print(tb1)
#     print("Total Label:", total, "TP:", array_TP, "FP:", array_FP, "FN:", total-array_TP)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', help='Path to directory containing images')
    parser.add_argument('--model_path', help='Path to model')
    parser.add_argument('--class_list', help='Path to CSV file listing class names (see README)')

    parser = parser.parse_args()
    print('Detecting Image, please wait.....')
    detect_image(parser.image_dir, parser.model_path, parser.class_list)

