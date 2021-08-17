import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Reduce frames in Trackingnet')
    parser.add_argument(
        'src_folder', type=str, help='root directory for the frames or videos')
    parser.add_argument(
        '--stride',
        type=int,
        default=10,
        help='temporal stride to reduce frames')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print(args)


if __name__ == '__main__':
    main()
