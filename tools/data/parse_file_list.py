import csv
import fnmatch
import glob
import json
import os
import os.path as osp


def parse_directory(path,
                    rgb_prefix='img_',
                    flow_x_prefix='flow_x_',
                    flow_y_prefix='flow_y_',
                    level=1):
    """Parse directories holding extracted frames from standard benchmarks.

    Args:
        path (str): Directory path to parse frames.
        rgb_prefix (str): Prefix of generated rgb frames name.
            default: 'img_'.
        flow_x_prefix (str): Prefix of generated flow x name.
            default: `flow_x_`.
        flow_y_prefix (str): Prefix of generated flow y name.
            default: `flow_y_`.
        level (int): Directory level for glob searching. Options are 1 and 2.
            default: 1.

    Returns:
        dict: frame info dict with video id as key and tuple(path(str),
            rgb_num(int), flow_x_num(int)) as value.
    """
    print(f'parse frames under directory {path}')
    if level == 1:
        # Only search for one-level directory
        def locate_directory(x):
            return osp.basename(x)

        frame_dirs = glob.glob(osp.join(path, '*'))

    elif level == 2:
        # search for two-level directory
        def locate_directory(x):
            return osp.join(osp.basename(osp.dirname(x)), osp.basename(x))

        frame_dirs = glob.glob(osp.join(path, '*', '*'))

    else:
        raise ValueError('level can be only 1 or 2')

    def count_files(directory, prefix_list):
        """Count file number with a given directory and prefix.

        Args:
            directory (str): Data directory to be search.
            prefix_list (list): List or prefix.

        Returns:
            list (int): Number list of the file with the prefix.
        """
        lst = os.listdir(directory)
        cnt_list = [len(fnmatch.filter(lst, x + '*')) for x in prefix_list]
        return cnt_list

    # check RGB
    frame_dict = {}
    for i, frame_dir in enumerate(frame_dirs):
        total_num = count_files(frame_dir,
                                (rgb_prefix, flow_x_prefix, flow_y_prefix))
        dir_name = locate_directory(frame_dir)

        num_x = total_num[1]
        num_y = total_num[2]
        if num_x != num_y:
            raise ValueError(f'x and y direction have different number '
                             f'of flow images in video directory: {frame_dir}')
        if i % 200 == 0:
            print(f'{i} videos parsed')

        frame_dict[dir_name] = (frame_dir, total_num[0], num_x)

    print('frame directory analysis done')
    return frame_dict


def parse_ucf101_splits(level):
    """Parse UCF-101 dataset into "train", "val", "test" splits.

    Args:
        level (int): Directory level of data. 1 for the single-level directory,
            2 for the two-level directory.

    Returns:
        list: "train", "val", "test" splits of UCF-101.
    """
    class_index_file = 'data/ucf101/annotations/classInd.txt'
    train_file_template = 'data/ucf101/annotations/trainlist{:02d}.txt'
    test_file_template = 'data/ucf101/annotations/testlist{:02d}.txt'

    with open(class_index_file, 'r') as fin:
        class_index = [x.strip().split() for x in fin]
    class_mapping = {x[1]: int(x[0]) - 1 for x in class_index}

    def line_to_map(line):
        """A function to map line string to vid and label.

        Args:
            line (str): A long directory path, which is a text path.

        Returns:
            tuple[str, str]: (vid, label), vid is the video id,
                label is the video label.
        """
        items = line.strip().split()
        vid = osp.splitext(items[0])[0]
        if level == 1:
            vid = osp.basename(vid)
            label = items[0]
        elif level == 2:
            vid = osp.join(osp.basename(osp.dirname(vid)), osp.basename(vid))
            label = class_mapping[osp.dirname(items[0])]
        return vid, label

    splits = []
    for i in range(1, 4):
        with open(train_file_template.format(i), 'r') as fin:
            train_list = [line_to_map(x) for x in fin]

        with open(test_file_template.format(i), 'r') as fin:
            test_list = [line_to_map(x) for x in fin]
        splits.append((train_list, test_list))

    return splits


def parse_sthv1_splits(level):
    """Parse Something-Something dataset V1 into "train", "val" splits.

    Args:
        level (int): Directory level of data. 1 for the single-level directory,
            2 for the two-level directory.

    Returns:
        list: "train", "val", "test" splits of Something-Something V1 dataset.
    """
    # Read the annotations
    # yapf: disable
    class_index_file = 'data/sthv1/annotations/something-something-v1-labels.csv'  # noqa
    # yapf: enable
    train_file = 'data/sthv1/annotations/something-something-v1-train.csv'
    val_file = 'data/sthv1/annotations/something-something-v1-validation.csv'
    test_file = 'data/sthv1/annotations/something-something-v1-test.csv'

    with open(class_index_file, 'r') as fin:
        class_index = [x.strip() for x in fin]
    class_mapping = {class_index[idx]: idx for idx in range(len(class_index))}

    def line_to_map(line, test_mode=False):
        items = line.strip().split(';')
        vid = items[0]
        if level == 1:
            vid = osp.basename(vid)
        elif level == 2:
            vid = osp.join(osp.basename(osp.dirname(vid)), osp.basename(vid))
        if test_mode:
            return vid
        else:
            label = class_mapping[items[1]]
            return vid, label

    with open(train_file, 'r') as fin:
        train_list = [line_to_map(x) for x in fin]

    with open(val_file, 'r') as fin:
        val_list = [line_to_map(x) for x in fin]

    with open(test_file, 'r') as fin:
        test_list = [line_to_map(x, test_mode=True) for x in fin]

    splits = ((train_list, val_list, test_list), )
    return splits


def parse_sthv2_splits(level):
    """Parse Something-Something dataset V2 into "train", "val" splits.

    Args:
        level (int): Directory level of data. 1 for the single-level directory,
            2 for the two-level directory.

    Returns:
        list: "train", "val", "test" splits of Something-Something V2 dataset.
    """
    # Read the annotations
    # yapf: disable
    class_index_file = 'data/sthv2/annotations/something-something-v2-labels.json'  # noqa
    # yapf: enable
    train_file = 'data/sthv2/annotations/something-something-v2-train.json'
    val_file = 'data/sthv2/annotations/something-something-v2-validation.json'
    test_file = 'data/sthv2/annotations/something-something-v2-test.json'

    with open(class_index_file, 'r') as fin:
        class_mapping = json.loads(fin.read())

    def line_to_map(item, test_mode=False):
        vid = item['id']
        if level == 1:
            vid = osp.basename(vid)
        elif level == 2:
            vid = osp.join(osp.basename(osp.dirname(vid)), osp.basename(vid))
        if test_mode:
            return vid
        else:
            template = item['template'].replace('[', '')
            template = template.replace(']', '')
            label = class_mapping[template]
            return vid, label

    with open(train_file, 'r') as fin:
        items = json.loads(fin.read())
        train_list = [line_to_map(item) for item in items]

    with open(val_file, 'r') as fin:
        items = json.loads(fin.read())
        val_list = [line_to_map(item) for item in items]

    with open(test_file, 'r') as fin:
        items = json.loads(fin.read())
        test_list = [line_to_map(item, test_mode=True) for item in items]

    splits = ((train_list, val_list, test_list), )
    return splits


def parse_mmit_splits():
    """Parse Multi-Moments in Time dataset into "train", "val" splits.

    Returns:
        list: "train", "val", "test" splits of Multi-Moments in Time.
    """

    # Read the annotations
    def line_to_map(x):
        vid = osp.splitext(x[0])[0]
        labels = [int(digit) for digit in x[1:]]
        return vid, labels

    csv_reader = csv.reader(open('data/mmit/annotations/trainingSet.csv'))
    train_list = [line_to_map(x) for x in csv_reader]

    csv_reader = csv.reader(open('data/mmit/annotations/validationSet.csv'))
    val_list = [line_to_map(x) for x in csv_reader]

    test_list = val_list  # not test for mit

    splits = ((train_list, val_list, test_list), )
    return splits


def parse_kinetics_splits(level):
    """Parse Kinetics-400 dataset into "train", "val", "test" splits.

    Args:
        level (int): Directory level of data. 1 for the single-level directory,
            2 for the two-level directory.

    Returns:
        list: "train", "val", "test" splits of Kinetics-400.
    """

    def convert_label(s, keep_whitespaces=False):
        """Convert label name to a formal string.

        Remove redundant '"' and convert whitespace to '_'.

        Args:
            s (str): String to be converted.
            keep_whitespaces(bool): Whether to keep whitespace. Default: False.

        Returns:
            str: Converted string.
        """
        if not keep_whitespaces:
            return s.replace('"', '').replace(' ', '_')
        else:
            return s.replace('"', '')

    def line_to_map(x, test=False):
        """A function to map line string to vid and label.

        Args:
            x (str): A single line from Kinetics-400 csv file.
            test (bool): Indicate whether the line comes from test
                annotation file.

        Returns:
            tuple[str, str]: (vid, label), vid is the video id,
                label is the video label.
        """
        if test:
            # vid = f'{x[0]}_{int(x[1]):06d}_{int(x[2]):06d}'
            vid = f'{x[1]}_{int(float(x[2])):06d}_{int(float(x[3])):06d}'
            label = -1  # label unknown
            return vid, label
        else:
            vid = f'{x[1]}_{int(float(x[2])):06d}_{int(float(x[3])):06d}'
            if level == 2:
                vid = f'{convert_label(x[0])}/{vid}'
            else:
                assert level == 1
            label = class_mapping[convert_label(x[0])]
            return vid, label

    train_file = 'data/kinetics400/annotations/kinetics_train.csv'
    val_file = 'data/kinetics400/annotations/kinetics_val.csv'
    test_file = 'data/kinetics400/annotations/kinetics_test.csv'

    csv_reader = csv.reader(open(train_file))
    # skip the first line
    next(csv_reader)

    labels_sorted = sorted(set([convert_label(row[0]) for row in csv_reader]))
    class_mapping = {label: i for i, label in enumerate(labels_sorted)}

    csv_reader = csv.reader(open(train_file))
    next(csv_reader)
    train_list = [line_to_map(x) for x in csv_reader]

    csv_reader = csv.reader(open(val_file))
    next(csv_reader)
    val_list = [line_to_map(x) for x in csv_reader]

    csv_reader = csv.reader(open(test_file))
    next(csv_reader)
    test_list = [line_to_map(x, test=True) for x in csv_reader]

    splits = ((train_list, val_list, test_list), )
    return splits


def parse_mit_splits():
    """Parse Moments in Time dataset into "train", "val" splits.

    Returns:
        list: "train", "val", "test" splits of Moments in Time.
    """
    # Read the annotations
    class_mapping = {}
    with open('data/mit/annotations/moments_categories.txt') as f_cat:
        for line in f_cat.readlines():
            cat, digit = line.rstrip().split(',')
            class_mapping[cat] = int(digit)

    def line_to_map(x):
        vid = osp.splitext(x[0])[0]
        label = class_mapping[osp.dirname(x[0])]
        return vid, label

    csv_reader = csv.reader(open('data/mit/annotations/trainingSet.csv'))
    train_list = [line_to_map(x) for x in csv_reader]

    csv_reader = csv.reader(open('data/mit/annotations/validationSet.csv'))
    val_list = [line_to_map(x) for x in csv_reader]

    test_list = val_list  # no test for mit

    splits = ((train_list, val_list, test_list), )
    return splits


def parse_hmdb51_split(level):
    train_file_template = 'data/hmdb51/annotations/trainlist{:02d}.txt'
    test_file_template = 'data/hmdb51/annotations/testlist{:02d}.txt'
    class_index_file = 'data/hmdb51/annotations/classInd.txt'

    def generate_class_index_file():
        """This function will generate a `ClassInd.txt` for HMDB51 in a format
        like UCF101, where class id starts with 1."""
        frame_path = 'data/hmdb51/rawframes'
        annotation_dir = 'data/hmdb51/annotations'

        class_list = sorted(os.listdir(frame_path))
        class_dict = dict()
        with open(class_index_file, 'w') as f:
            content = []
            for class_id, class_name in enumerate(class_list):
                # like `ClassInd.txt` in UCF-101, the class_id begins with 1
                class_dict[class_name] = class_id + 1
                cur_line = ' '.join([str(class_id + 1), class_name])
                content.append(cur_line)
            content = '\n'.join(content)
            f.write(content)

        for i in range(1, 4):
            train_content = []
            test_content = []
            for class_name in class_dict:
                filename = class_name + f'_test_split{i}.txt'
                filename_path = osp.join(annotation_dir, filename)
                with open(filename_path, 'r') as fin:
                    for line in fin:
                        video_info = line.strip().split()
                        video_name = video_info[0]
                        if video_info[1] == '1':
                            target_line = ' '.join([
                                osp.join(class_name, video_name),
                                str(class_dict[class_name])
                            ])
                            train_content.append(target_line)
                        elif video_info[1] == '2':
                            target_line = ' '.join([
                                osp.join(class_name, video_name),
                                str(class_dict[class_name])
                            ])
                            test_content.append(target_line)
            train_content = '\n'.join(train_content)
            test_content = '\n'.join(test_content)
            with open(train_file_template.format(i), 'w') as fout:
                fout.write(train_content)
            with open(test_file_template.format(i), 'w') as fout:
                fout.write(test_content)

    if not osp.exists(class_index_file):
        generate_class_index_file()

    with open(class_index_file, 'r') as fin:
        class_index = [x.strip().split() for x in fin]
    class_mapping = {x[1]: int(x[0]) - 1 for x in class_index}

    def line_to_map(line):
        items = line.strip().split()
        vid = osp.splitext(items[0])[0]
        if level == 1:
            vid = osp.basename(vid)
        elif level == 2:
            vid = osp.join(osp.basename(osp.dirname(vid)), osp.basename(vid))
        label = class_mapping[osp.dirname(items[0])]
        return vid, label

    splits = []
    for i in range(1, 4):
        with open(train_file_template.format(i), 'r') as fin:
            train_list = [line_to_map(x) for x in fin]

        with open(test_file_template.format(i), 'r') as fin:
            test_list = [line_to_map(x) for x in fin]
        splits.append((train_list, test_list))

    return splits


def parse_davis2017_splits():
    """Parse DAVIS2017 dataset into "train", "val" splits.

    Returns:
        list: "train", "val", "test" splits of Moments in Time.
    """
    # Read the annotations

    with open('data/davis/DAVIS/ImageSets/2017/train.txt') as f:
        train_list = [(vid.rstrip(), idx)
                      for idx, vid in enumerate(f.readlines())]
    with open('data/davis/DAVIS/ImageSets/2017/val.txt') as f:
        val_list = [(vid.rstrip(), idx)
                    for idx, vid in enumerate(f.readlines())]
    with open('data/davis/DAVIS/ImageSets/2017/test-dev.txt') as f:
        test_list = [(vid.rstrip(), idx)
                     for idx, vid in enumerate(f.readlines())]

    splits = ((train_list, val_list, test_list), )
    return splits


def parse_jhmdb_splits():
    """Parse DAVIS2017 dataset into "train", "val" splits.

    Returns:
        list: "train", "val", "test" splits of Moments in Time.
    """
    # Read the annotations
    import pickle
    with open('data/jhmdb/JHMDB/JHMDB-GT.pkl', 'rb') as f:
        # u = pickle._Unpickler(f)
        # u.encoding = 'latin1'
        # gt_info = u.load()
        gt_info = pickle.load(f, encoding='latin1')
    train_list = gt_info['train_videos']
    test_list = gt_info['test_videos']
    for i in range(len(train_list)):
        for j in range(len(train_list[i])):
            train_list[i][j] = (train_list[i][j], j)
    for i in range(len(test_list)):
        for j in range(len(test_list[i])):
            test_list[i][j] = (test_list[i][j], j)

    splits = ((train_list[0], test_list[0]), (train_list[1], test_list[1]),
              (train_list[2], test_list[2]))
    return splits


def parse_vip_splits():
    """Parse DAVIS2017 dataset into "train", "val" splits.

    Returns:
        list: "train", "val", "test" splits of Moments in Time.
    """
    # Read the annotations
    with open('data/vip/VIP_Fine/lists/train_videos.txt') as f:
        train_list = [(vid.rstrip(), idx)
                      for idx, vid in enumerate(f.readlines())]
    with open('data/vip/VIP_Fine/lists/val_videos.txt') as f:
        val_list = [(vid.rstrip(), idx)
                    for idx, vid in enumerate(f.readlines())]
    with open('data/vip/VIP_Fine/lists/test_videos.txt') as f:
        test_list = [(vid.rstrip(), idx)
                     for idx, vid in enumerate(f.readlines())]

    splits = ((train_list, val_list, test_list), )
    return splits
