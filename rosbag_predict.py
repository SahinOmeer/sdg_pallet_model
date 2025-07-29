#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run pallet model inference on frames stored in a ROS bag file.

This script requires ROS and ``cv_bridge`` to be installed in your environment.
It is intended for ROS1 bags and can be executed inside Docker if desired.
"""

import argparse
import cv2
import torch
import rosbag
from cv_bridge import CvBridge

import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "engine",
        type=str,
        help="The file path of the TensorRT engine."
    )
    parser.add_argument(
        "bag",
        type=str,
        help="The path to the input rosbag file."
    )
    parser.add_argument(
        "--image-topic",
        type=str,
        default="/camera/image_raw",
        help="Image topic to read from the bag."
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default="rosbag_output.mp4",
        help="Path of the output annotated video."
    )
    parser.add_argument(
        "--inference-size",
        type=str,
        default="512x512",
        help="Height and width that the image is resized to for inference."
    )
    parser.add_argument(
        "--peak-window",
        type=str,
        default="7x7",
        help="Window size used for finding local peaks."
    )
    parser.add_argument(
        '--peak-threshold',
        type=float,
        default=0.5,
        help="The heatmap threshold used for peak extraction."
    )
    parser.add_argument(
        '--line-thickness',
        type=int,
        default=1,
        help="Thickness of drawn boxes."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    inference_size = tuple(int(x) for x in args.inference_size.split('x'))
    peak_window = tuple(int(x) for x in args.peak_window.split('x'))

    offset_grid = utils.make_offset_grid(inference_size).to("cuda")

    model = utils.load_trt_engine_wrapper(
        args.engine,
        input_names=["input"],
        output_names=["heatmap", "vectormap"]
    )

    bag = rosbag.Bag(args.bag)
    bridge = CvBridge()

    writer = None
    stream = torch.cuda.Stream()

    with torch.no_grad():
        for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
            image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            orig_h, orig_w = image.shape[:2]

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(args.output_video, fourcc, 30, (orig_w, orig_h))

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_proc, _, _ = utils.pad_resize(image_rgb, inference_size)

            x = utils.format_bgr8_image(image_proc)
            x = x.to("cuda", non_blocking=True)

            with torch.cuda.stream(stream):
                heatmap, vectormap = model(x)

                keypointmap = utils.vectormap_to_keypointmap(
                    offset_grid,
                    vectormap
                )

                peak_mask = utils.find_heatmap_peak_mask(
                    heatmap,
                    peak_window,
                    args.peak_threshold
                )
            stream.synchronize()

            keypoints = keypointmap[0][peak_mask[0, 0]].cpu().numpy()

            vis_image = utils.draw_box(
                image,
                keypoints,
                thickness=args.line_thickness
            )

            writer.write(vis_image)

    if writer is not None:
        writer.release()
    bag.close()


if __name__ == "__main__":
    main()
