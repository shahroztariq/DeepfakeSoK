import argparse
import os
import numpy as np
import cv2
import utils.shared as shared
from os.path import join
from utils.landmark_utils import detect_frames_track


def detect_track(input_path, video):
    vidcap = cv2.VideoCapture(join(input_path, video))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        success, image = vidcap.read()
        if success:
            frames.append(image)
        else:
            break

    raw_data = detect_frames_track(frames, video, fps)

    vidcap.release()
    return np.array(raw_data)


def main(args):

    input_path = args.input_path
    output_path = args.output_path

    shared.use_visualization = args.visualize
    shared.visualize_path = args.visualize_path
    shared.log_file = args.log_file
    shared.face_detector_selection = args.fd

    """
    Prepare the environment
    """
    assert os.path.exists(input_path), "Input path does not exist."

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if shared.use_visualization:
        print("Settings: Visualize the extraction results.")
        if not os.path.exists(shared.visualize_path):
            os.makedirs(shared.visualize_path)
    else:
        print("Settings: NOT visualize the extraction results.")

    videos = os.listdir(input_path)
    videos.sort()
    count_videos = 0
    for video in videos:
        if video.startswith('.') or (not video.endswith('mp4')):
            continue

        video_name = video.split('.')[0]
        if os.path.exists(join(output_path, video_name + ".txt")):
            print(video_name + " exist, skipped")
            continue
        print("Extract landmarks from {}.".format(video))
        raw_data = detect_track(input_path, video)

        if len(raw_data) == 0:
            print("No face detected", video)
        else:
            np.savetxt(join(output_path, video_name + ".txt"), raw_data, fmt='%1.5f')
        print("Landmarks data saved!")
        count_videos += 1
    return count_videos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract landmarks sequences from input videos.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input_path', type=str, default=None,
                        help="Input videos path (folder)"
                        )
    parser.add_argument('-o', '--output_path', type=str, default='./landmarks/',
                        help="Output landmarks(.txt) path (folder)"
                        )
    parser.add_argument('-v', '--visualize', action='store_true', default=False,
                        help="If visualize the extraction results."
                        )
    parser.add_argument('--visualize_path', type=str, default='./visualize/',
                        help="Visualize videos path (folder)."
                        )
    parser.add_argument('-l', '--log_file', type=str, default='landmark_logs.txt',
                        help="The log file's name (generated under the /demo by default)."
                        )
    parser.add_argument('--fd', type=str, default='blazeface',
                        choices=['blazeface', 'retinaface'],
                        help="Select the face detector. (blazeface or retinaface)"
                        )
    args = parser.parse_args()
    if args.input_path is not None:
        main(args)
    else:
        print("Runing upon setting")
        for data_name in (  "LightWeight", "real"): #"DeepFaceLab", "Dfaker", "Faceswap", "FOM_Animation", "FOM_Faceswap", "FSGAN", "LightWeight", "real"
            args.input_path =   f"/media/data2/binh/CSIRO/generated_data_single/{data_name}/"
            args.output_path = f'./landmarks_new/{data_name}'
            args.visualize = False
            os.makedirs(args.output_path, exist_ok=True)
            count_videos = main(args)
            print(f"Total video of {data_name}: ", count_videos)
