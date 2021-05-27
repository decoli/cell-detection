import os
import argparse
import glob
import random

def make_trainval_list(args):
    # get annotation list
    annotation_path = os.path.join(args.root, '*.xml')
    annotation_list = glob.glob(annotation_path)
    random.shuffle(annotation_list)

    # make .txt
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    with open(args.output_file, 'a') as f:
        for each_annotation in annotation_list:
            name_ex = os.path.basename(each_annotation)
            f.write('{}\n'.format(os.path.splitext(name_ex)[0]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-R', '--root', required=True)
    parser.add_argument('-O', '--output_file', default='output/trainval.txt')
    args = parser.parse_args()

    make_trainval_list(args)
    print('---\n{} complete.'.format(os.path.basename(__file__)))
