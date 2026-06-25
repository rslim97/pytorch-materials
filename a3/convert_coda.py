import json
import os
from tqdm import tqdm
import cv2
import numpy as np
import re
import shutil


def visualize_coco(annotation_file, data_root):
    """
    annotation_file : a coco format json annotation file
    data_root : the path where the images are saved
    """
    with open(annotation_file, "r") as f:

        data = json.load(f)

    # id_map = {}
    for i, category in enumerate(data["categories"]):
        print(i, category)
        # id_map[category['id']] = i  # remap

    cnt = 0
    for img in tqdm(data["images"]):
        img_id = img["id"]
        # img_width = img['width']
        # img_height = img['height']

        img_name = img["file_name"]

        print("image_name", img_name)
        bbox_list = []
        label_list = []

        annotations = [ann for ann in data["annotations"] if ann["image_id"] == img_id]

        for ann in annotations:
            print("ann", ann)

            category_id = ann["category_id"]
            label_list.append(category_id)

            bbox = ann["bbox"]
            bbox_list.append(bbox)
            # x, y, w, h = ann['bbox']  # COCO has TLWH bbox format

        print("bbox_list", bbox_list)
        if cnt == 1:
            break
        cnt += 1

        fname = os.path.join(data_root, img_name)
        print(fname)
        img = cv2.imread(fname)
        # Draw bounding boxes
        for bbox, label in zip(bbox_list, label_list):
            x_tl, y_tl, w, h = bbox
            tl = (int(x_tl), int(y_tl))
            br = (int(x_tl + w), int(y_tl + h))
            cv2.rectangle(img, tl, br, (0, 255, 0), 1)
            cv2.putText(
                img, str(label), tl, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1
            )
        cv2.imshow(fname, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def visualize_coda(annotation_file, data_root):

    with open(annotation_file, "r") as f:
        data = json.load(f)

    keys = [cat["id"] for cat in data["categories"]]
    values = [
        [
            np.random.randint(125, 225),
            np.random.randint(125, 225),
            np.random.randint(125, 225),
        ]
        for _ in range(len(keys))
    ]
    color_map = dict(zip(keys, values))

    cnt = 0
    for ann in data["annotations"]:
        img_fname = ann["image_name"]
        fname = os.path.join(data_root, img_fname)
        img = cv2.imread(fname)
        for bbox, label in zip(ann["bbox"], ann["category_id"]):
            x_tl, y_tl, w, h = bbox
            tl = (int(x_tl), int(y_tl))
            br = (int(x_tl + w), int(y_tl + h))
            color = color_map[label]
            color[label % 3] = 0
            cv2.rectangle(img, tl, br, color_map[label], 2)
            cv2.putText(
                img,
                CAR_CLASSES[label - 1],
                tl,
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                color_map[label],
                2,
            )
        if cnt > 85:
            cv2.imshow(fname, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if cnt == 100:
            break
        cnt += 1


def combine_dataset(data_dir, output_dir):
    """
    Returns :
    images: list<dict>  # Combined images
    annotations: list<dict>  # Combined bounding box annotations
    categories: list<dict>  # Object class
    """
    # output_dir = 'coda_small'
    # Source datasets to be combined
    # src1 = 'CODA-val-1500/CODA/base-val-1500'  # Corner cases
    src2 = "CODA2022-test"
    src3 = "CODA2022-val"
    # root1 = os.path.join(dataset1, 'images')
    # root2 = os.path.join(dataset2, 'images')
    # root3 = os.path.join(dataset3, 'images')
    # annotation_file1 = os.path.join(dst1, 'corner_case.json')  # Only annotations for corner cases
    annotation_file2 = os.path.join(data_dir, src2, "annotations.json")
    annotation_file3 = os.path.join(data_dir, src3, "annotations.json")

    data = dict()
    if not os.path.isdir(output_dir):
        os.makedirs(os.path.join(output_dir, "images"))

    # with open(annotation_file1, 'r') as f:
    #     data1 = json.load(f)
    with open(annotation_file2, "r") as f:
        data2 = json.load(f)
    with open(annotation_file3, "r") as f:
        data3 = json.load(f)

    # print(type(data1))
    print(type(data2))
    print(type(data3))

    # Save source directory of images
    # for img in data1['images']:
    #     img['source_dir'] = dst1
    for img in data2["images"]:
        img["source_dir"] = src2
    for img in data3["images"]:
        img["source_dir"] = src3

    print(data2.keys())  # 'images', 'categories', 'annotations'
    print(type(data2["images"]))
    print(type(data2["annotations"]))
    print(data2["images"][:5])
    print(data2["annotations"][:5])
    print(data2["categories"])

    # cnt=0
    def f(dat, offset):
        cnt = 0
        unique_images = set()
        for img in dat["images"]:
            if img["id"] not in unique_images:
                unique_images.add(img["id"])
            else:
                continue
            # Save old id
            img["old_id"] = img["id"]
            # Update with new id
            img["id"] += offset
            # Save old image file name
            img["old_file_name"] = img["file_name"]
            # Update image file_name
            img["file_name"] = str(img["id"]) + ".jpg"
            # Update annotations
            img_annotations = [
                ann for ann in dat["annotations"] if ann["image_id"] == img["old_id"]
            ]
            for ann in img_annotations:
                ann["old_image_id"] = img["old_id"]
                ann["image_id"] = img["id"]
            cnt += 1
            # Copy image
            img_src = os.path.join(
                data_dir, img["source_dir"], "images", img["old_file_name"]
            )
            img_dst = os.path.join(output_dir, "images", img["file_name"])
            shutil.copy2(img_src, img_dst)
        return cnt

    # # unique_images = set()
    # # Modify image ids
    # def remap_image_id(data, img_offset, ann_offset):
    #     count_imgs = 0; count_anns = 0
    #     img_id_map = {}
    #     for img in data['images']:
    #         # if img['file_name'] in unique_images:
    #         #     print(f"Warning: Duplicate image {img['file_name']} from {img['source_dir']}. Skipping." )
    #         #     continue
    #         #     # img_id_map[img['id']]
    #         # unique_images.add(img['file_name'])

    #         # Store old img_id
    #         old_img_id = img['id']
    #         old_img_file_name = img['file_name']

    #         # Update img_id
    #         img['id'] += img_offset
    #         # Update map
    #         img_id_map[old_img_id] = img['id']
    #         # New image file name
    #         img['file_name'] = str(img['id']) + '.jpg'

    #         # Overwrite old with new image file name
    #         new_img_file_name = img['file_name']

    #         # Copy and rename all images with unique names to remove duplicates,
    #         # e.g. img 001.jpg in train directory and another 001.jpg in val directory
    #         # but of different scenes
    #         src = os.path.join(img['source_dir'], 'images', old_img_file_name)
    #         dst = os.path.join(output_dir, 'images', new_img_file_name)
    #         # Copy image to combined
    #         shutil.copy2(src, dst)
    #         count_imgs += 1

    #     for ann in data['annotations']:
    #         ann['id'] += ann_offset
    #         ann['image_id'] = img_id_map[ann['image_id']]
    #         count_anns += 1

    #     return count_imgs, count_anns

    # count_imgs2, count_anns2 = remap_image_id(data2, 0, 0)
    # count_imgs3, count_anns3 = remap_image_id(data3, count_imgs2, count_anns2)

    offset = f(data2, 0)
    _ = f(data3, offset)

    # all_images = data2['images'] + data3['images']
    # all_annotations = data2['annotations'] + data3['annotations']
    # categories = data2['categories']
    images_combined = data2["images"] + data3["images"]
    annotations_combined = data2["annotations"] + data3["annotations"]
    categories = data2["categories"]

    # # print(type(all_images))
    # # print(type(all_annotations))
    # # print(type(categories))
    # # print(type(all_images[0]))
    # # print(type(all_annotations[0]))
    # # print(type(categories[0]))

    data["images"] = images_combined
    data["annotations"] = annotations_combined
    data["categories"] = categories
    with open(f"{output_dir}/annotations_combined.json", "w") as f:
        json.dump(data, f)

    # all_data['images'] = all_images
    # all_data['annotations'] = all_annotations
    # all_data['categories'] = categories
    # with open(f'{output_dir}/combined_annotations.json', 'w') as f:
    #     json.dump(all_data, f)

    print(f"Total no. of images: {len(images_combined)}")
    print(f"Total no. of bounding box annotations: {len(annotations_combined)}")
    print(f"No. of categories, {len(categories)}")
    # return all_images, all_annotations, categories
    return images_combined, annotations_combined, categories


# import shutil

# CAR_CLASSES = ["car", "pedestrian", "cyclist", "truck", "tram"]
# CAR_CLASSES = ['Pedestrian', 'Cyclist', 'Car', 'Truck', 'Tram']
CAR_CLASSES = [
    "pedestrian",
    "cyclist",
    "car",
    "truck",
    "tram",
]  # in order of sequence in data["categories"]


def filter_dataset(images, annotations, categories):

    data = dict()
    # Filter out nuscenes images and annotations
    data["annotations"] = []
    data["categories"] = []
    cat_id_map = {}
    # cat_id_map_2 = {}
    cat_id = 1  # 0 Background is reserved for background only
    # Filter categories of interest and remap to new category ids
    for cat in categories:
        print('cat["name"]', cat["name"])
        if cat["name"] in CAR_CLASSES:
            # dict[old] = new
            cat_id_map[cat["id"]] = cat_id
            # cat_id_map_2[cat_id] = cat["id"]
            data["categories"].append(
                {
                    "name": cat["name"],
                    "id": cat_id,
                    "supercategory": cat["supercategory"],
                }
            )
            cat_id += 1
    # print(all_data['categories'])
    print("cat_id_map", cat_id_map)

    # cnt = 0
    for img in tqdm(images):
        img_id = img["id"]
        img_fname = img["file_name"]
        src_dir = img["source_dir"]
        if re.search("kitti", img_fname) or re.search("nuscenes", img_fname):
            continue
        # Filter
        img_annotations = [ann for ann in annotations if ann["image_id"] == img_id]
        """
        old_id : ann['category_id']
        new_id : cat_id_map[ann['category_id']]
        """
        category_ids = [
            cat_id_map[ann["category_id"]]
            for ann in img_annotations
            if ann["category_id"] in cat_id_map.keys()
        ]
        bboxes = [
            ann["bbox"]
            for ann in img_annotations
            if ann["category_id"] in cat_id_map.keys()
        ]

        # Plot image for debug
        # fname = '000' + img_fname
        # print('fname', fname)
        # img = cv2.imread(os.path.join('a3', 'data', src_dir, 'images', fname))
        # cv2.rectangle()
        # keys = [c for c in category_ids]
        # values = [
        #     [
        #         np.random.randint(125, 225),
        #         np.random.randint(125, 225),
        #         np.random.randint(125, 225),
        #     ]
        #     for _ in range(len(keys))
        # ]
        # color_map = dict(zip(keys, values))

        # for bbox, label in zip(bboxes, category_ids):
        #     x_tl, y_tl, w, h = bbox
        #     tl = (int(x_tl), int(y_tl))
        #     br = (int(x_tl + w), int(y_tl + h))
        #     color = color_map[label]
        #     color[label % 3] = 0
        #     cv2.rectangle(img, tl, br, color_map[label], 2)
        #     cv2.putText(img, CAR_CLASSES[label-1], tl, cv2.FONT_HERSHEY_COMPLEX, 0.5, color_map[label], 2)
        # cv2.imshow('image', img)
        # cv2.imwrite('save_debug.jpg', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # fname = os.path.join(img['source_dir'], 'images', img_name)
        # print(fname)
        # image = cv2.imread(fname)
        # for bbox in bboxes:
        #     x_tl, y_tl, w, h = bbox
        #     tl = (int(x_tl), int(y_tl))
        #     br = (int(x_tl + w), int(y_tl + h))
        #     cv2.rectangle(image, tl, br, (0, 0, 255), 1)
        # cv2.imshow(fname, image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        data["annotations"].append(
            {
                "image_name": img_fname,
                "category_id": category_ids,
                "bbox": bboxes,
                "source_dir": src_dir,
            }
        )
        # if cnt == 1000:
        #     break
        # cnt+=1

    return data


def save_and_split(data, split, src_dir, dst_dir):
    # output_dir = 'coda_small'

    for ann in data["annotations"]:
        # Source directory for images
        # root = os.path.join(ann['source_dir'], 'images')

        # shutil.copy
        img_fname = ann["image_name"]
        img_src = os.path.join(src_dir, img_fname)
        img_dst = os.path.join(dst_dir, split, "images", img_fname)
        shutil.copy(img_src, img_dst)

    # Save annotations
    with open(f"{dst_dir}/annotations/instance_{split}.json", "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(proj_dir, "data")
    output_dir = os.path.join(proj_dir, "coda_combined")
    dst_dir = os.path.join(proj_dir, "coda_small")
    # coco_annotation_file = "CODA-val-1500/CODA/base-val-1500/corner_case.json"
    # img_dir = 'CODA-val-1500/CODA/base-val-1500/images'

    coco_annotation_file = os.path.join(data_dir, "CODA2022-val/annotations.json")
    img_root = os.path.join(data_dir, "CODA2022-val/images")

    # coco_annotation_file = "CODA2022-test/annotations.json"
    # img_dir = 'CODA2022-test/images'

    visualize_coco(annotation_file=coco_annotation_file, data_root=img_root)

    images, annotations, categories = combine_dataset(data_dir, output_dir)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    else:
        shutil.rmtree(dst_dir)
        os.makedirs(dst_dir)

    for split in ["annotations", "train", "val", "test"]:
        os.mkdir(os.path.join(dst_dir, split))
        if split != "annotations":
            os.mkdir(os.path.join(dst_dir, split, "images"))

    data = filter_dataset(images, annotations, categories)

    print(type(data))
    print(data.keys())  # 'annotations', 'categories'
    print(data["annotations"][:5])
    print(data["categories"])

    np.random.seed(42)
    np.random.shuffle(data["annotations"])

    # Total no. of samples
    N = len(data["annotations"])
    train_data = {
        "annotations": data["annotations"][: int(0.5 * N)],
        "categories": data["categories"],
    }
    val_data = {
        "annotations": data["annotations"][int(0.5 * N) : int(0.75 * N)],
        "categories": data["categories"],
    }
    test_data = {
        "annotations": data["annotations"][int(0.75 * N) :],
        "categories": data["categories"],
    }

    img_root = os.path.join(output_dir, "images")
    save_and_split(train_data, "train", img_root, dst_dir)
    save_and_split(val_data, "val", img_root, dst_dir)
    save_and_split(test_data, "test", img_root, dst_dir)

    coda_annotation_file = os.path.join(dst_dir, "annotations/instance_train.json")
    coda_img_root = os.path.join(dst_dir, "train/images")
    visualize_coda(annotation_file=coda_annotation_file, data_root=coda_img_root)
