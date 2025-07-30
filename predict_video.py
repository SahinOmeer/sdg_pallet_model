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

import argparse
import utils
import cv2
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "engine",
        type=str,
        help="The file path of the TensorRT engine."
    )
    parser.add_argument(
        "video",
        type=str,
        help="The file path of the MP4 video provided as input for inference."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="The path to output the inference visualization video."
    )
    parser.add_argument(
        "--inference-size",
        type=str,
        default="512x512",
        help="The height and width that frames are resized to for inference."
             " Denoted as (height)x(width)."
    )
    parser.add_argument(
        "--peak-window",
        type=str,
        default="7x7",
        help="The size of the window used when finding local peaks. Denoted as "
             " (window_height)x(window_width)."
    )
    parser.add_argument(
        '--peak-threshold',
        type=float,
        default=0.5,
        help="The heatmap threshold to use when finding peaks.  Values must be "
             " larger than this value to be considered peaks."
    )
    parser.add_argument(
        '--line-thickness',
        type=int,
        default=1,
        help="The line thickness for drawn boxes"
    )

    args = parser.parse_args()

    inference_size = tuple(int(x) for x in args.inference_size.split('x'))
    peak_window = tuple(int(x) for x in args.peak_window.split('x'))

    if args.output is None:
        output_path = '.'.join(args.video.split('.')[:-1]) + "_output.mp4"
    else:
        output_path = args.output

    offset_grid = utils.make_offset_grid(inference_size).to("cuda")

    model = utils.load_trt_engine_wrapper(
        args.engine,
        input_names=["input"],
        output_names=["heatmap", "vectormap"]
    )

    cap = cv2.VideoCapture(args.video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_resized, _, _ = utils.pad_resize(image, inference_size)

            x = utils.format_bgr8_image(image_resized)
            x = x.to("cuda")

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

            keypoints = keypointmap[0][peak_mask[0, 0]]

            vis_frame = utils.draw_box(
                image_resized,
                keypoints,
                color=(118, 186, 0),
                thickness=args.line_thickness
            )

            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            vis_frame = cv2.resize(vis_frame, (width, height))
            writer.write(vis_frame)

    cap.release()
    writer.release()
