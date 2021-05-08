import argparse
import glob
import os


def rename(args):
    annotation_path = os.path.join(args.annotation_root, '*.xml')
    annotation_path = glob.glob(annotation_path)

    for i, each_annotation_path in enumerate(annotation_path):
        name_ex = os.path.basename(each_annotation_path)
        img_path = os.path.join(
            args.image_root, '{img_name}.jpg'.format(
                img_name=os.path.splitext(name_ex)[0]))

        if os.path.exists(img_path):
            os.rename(
                each_annotation_path,
                os.path.join(args.annotation_root, '{i}.xml'.format(i=i)))
            os.rename(
                img_path,
                os.path.join(args.image_root, '{i}.jpg'.format(i=i)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--image_root', required=True)
    parser.add_argument('-A', '--annotation_root', required=True)
    args = parser.parse_args()

    rename(args)
    print("complete.")
