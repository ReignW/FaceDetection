import argparse
import os
import math
from time import time
from PIL import Image
import align.detect_face as detect_face
import cv2
import dlib
import numpy as np
import tensorflow as tf
from lib.face_utils import judge_side_face
from lib.utils import Logger, mkdir
from project_root_dir import project_dir
from src.sort import Sort
from load_model.tensorflow_loader import load_tf_model, tf_inference
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression

d_detector = dlib.get_frontal_face_detector()
d_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')



logger = Logger()

# 开始人脸对齐

sess, graph = load_tf_model('models/face_mask_detection.pb')
# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}

# 人脸对齐方法
def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(160, 160),
              draw_result=True,
              show_result=True
              ):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    '''

    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)


    # 输出回归框和人脸得分
    y_bboxes_output, y_cls_output = tf_inference(sess, graph, image_exp)

    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # 对单个人脸进行非极大抑制
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            # cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    # if show_result:
    #     Image.fromarray(image).show()

    return output_info


def main():
    global colours, img_size
    args = parse_args()
    videos_dir = args.videos_dir
    output_path = args.output_path
    no_display = args.no_display
    detect_interval = args.detect_interval  # 间隔一帧检测一次
    margin = args.margin  # 脸边距（默认10）
    scale_rate = args.scale_rate  # 检测图像的尺寸设置
    show_rate = args.show_rate  # 展示图像的尺寸设置
    face_score_threshold = args.face_score_threshold  # 人脸判别阈值

    mkdir(output_path)
    # for display
    if not no_display:
        colours = np.random.rand(32, 3)

    # 初始化追踪器
    tracker = Sort()  # create instance of the SORT tracker

    logger.info('Start track and extract......')



    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                                              log_device_placement=False)) as sess:

            for filename in os.listdir(videos_dir):
                logger.info('All files:{}'.format(filename))
            for filename in os.listdir(videos_dir):
                suffix = filename.split('.')[1]
                if suffix != 'mp4' and suffix != 'avi':  # you can specify more video formats if you need
                    continue
                video_name = os.path.join(videos_dir, filename)
                directoryname = os.path.join(output_path, filename.split('.')[0])
                logger.info('Video_name:{}'.format(video_name))
                cam = cv2.VideoCapture(video_name)
                ret, frame = cam.read()
                # print(frame.shape)
                frame_size = list(frame.shape[:2])
                frame_size[0],frame_size[1] = math.ceil(frame_size[1]*0.7),math.ceil(frame_size[0]*0.7)
                frame_size = tuple(frame_size)
                # print(frame_size)
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                videoWriter = cv2.VideoWriter(str(filename.split('.')[0]) + '.avi', fourcc, 24,
                                              frame_size)  # 最后一个是保存图片的尺寸

                c = 0

                while True:
                    try:
                        final_faces = []
                        addtional_attribute_list = []
                        ret, frame = cam.read()

                        if not ret:
                            logger.warning("ret false")
                            break
                        if frame is None:
                            logger.warning("frame drop")
                            break

                        frame = cv2.resize(frame, (0, 0), fx=scale_rate, fy=scale_rate)
                        r_g_b_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if c % detect_interval == 0:
                            img_size = np.asarray(frame.shape)[0:2]


                            # 获取人脸框（采用项目二算法,输出结果包括：带没带口罩，人脸分数，坐标）
                            all_faces = np.array(inference(r_g_b_frame, show_result=True, target_shape=(260, 260)))

                            faces_t = all_faces[...,1:]
                            faces = faces_t[:,[1,2,3,4,0]]
                            mask_detect = all_faces[...,0]
                            # print(mask_detect,'mask')


                            # 获取5个关键点（采用dlib，获取5个，分别是眼睛×2，鼻子×1，嘴巴×2。与mtcnn取点一致）


                            m = all_faces[..., 2:]  # 输出脸框进行切片，使得输出仅包含连框坐标
                            points = np.ones([10,1])  # 构建数组用于存放5个关键点坐标，并与模型结构保持一致
                            for ii in m:
                                # print(i)

                                rects = [dlib.rectangle(int(ii[0]), int(ii[1]), int(ii[2]), int(ii[3]))]
                                landmarks = np.matrix([[p.x, p.y] for p in d_predictor(r_g_b_frame, rects[0]).parts()])
                                # print(landmarks)
                                key_landmark = []  # 5个关键点分别为眼睛×2，一鼻子×1，两个嘴巴*2
                                left_eye = landmarks[36].getA()[0]
                                right_eye = landmarks[42].getA()[0]
                                nose = landmarks[29].getA()[0]
                                left_mouth = landmarks[48].getA()[0]
                                right_mouth = landmarks[54].getA()[0]
                                ans = np.array((left_eye, right_eye, nose, left_mouth, right_mouth))
                                ans_1 = ans[..., 0]
                                ans_2 = ans[..., 1]
                                bb = np.array([np.append(ans_1, ans_2)]).T
                                points = np.concatenate((points, bb), axis=1)

                            points = np.delete(points, 0, axis=1)
                            face_sums = faces.shape[0]

                            if face_sums > 0:
                                face_list = []
                                for i, item in enumerate(faces):

                                    score = round(faces[i, 4], 6)
                                    if score > face_score_threshold:
                                        det = np.squeeze(faces[i, 0:4])

                                        # face rectangle
                                        det[0] = np.maximum(det[0] - margin, 0)
                                        det[1] = np.maximum(det[1] - margin, 0)
                                        det[2] = np.minimum(det[2] + margin, img_size[1])
                                        det[3] = np.minimum(det[3] + margin, img_size[0])
                                        face_list.append(item)

                                        # face cropped
                                        bb = np.array(det, dtype=np.int32)

                                        # use 5 face landmarks  to judge the face is front or side
                                        squeeze_points = np.squeeze(points[:, i])

                                        tolist = squeeze_points.tolist()
                                        facial_landmarks = []
                                        for j in range(5):
                                            item = [tolist[j], tolist[(j + 5)]]
                                            facial_landmarks.append(item)
                                        if args.face_landmarks:
                                            for (x, y) in facial_landmarks:
                                                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

                                        # for (x, y) in facial_landmarks:
                                        #     cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

                                        cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :].copy()

                                        dist_rate, high_ratio_variance, width_rate = judge_side_face(
                                            np.array(facial_landmarks))
                                        # print('facial_land',facial_landmarks)

                                        # face addtional attribute(index 0:face score; index 1:0 represents front face and 1 for side face )
                                        item_list = [cropped, score, dist_rate, high_ratio_variance, width_rate]
                                        addtional_attribute_list.append(item_list)  # 数据结构5维，第一维是numpy对象
                                        # print(addtional_attribute_list)
                                        #
                                        # print(np.shape(addtional_attribute_list))
                                    else:
                                        mask_detect[i] = 9  # 将不满足的脸框口罩检测指数置为9



                                final_faces = np.array(face_list)
                                # print(final_faces)
                        # print(mask_detect[mask_detect[:]<7])
                        # 将mask参数带入对象中
                        tmp_2 = mask_detect[mask_detect[:] < 7]
                        tmp_2 = np.array([tmp_2]).T
                        addtional_attribute_list = np.concatenate((addtional_attribute_list,tmp_2),axis=1)

                        # print(np.shape(addtional_attribute_list))

                        trackers = tracker.update(final_faces, img_size, directoryname, addtional_attribute_list, detect_interval,mask_detect)

                        c += 1


                        for d in trackers:
                            # print(d)
                            # print(d[4])
                            if not no_display:
                                d = d.astype(np.int32)
                                cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), colours[d[4] % 32, :] * 255, 3)
                                if final_faces != []:
                                    cv2.putText(frame, 'ID : %d  DETECT' % (d[4]), (d[0] - 10, d[1] - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.75,
                                                colours[d[4] % 32, :] * 255, 2)
                                    cv2.putText(frame, 'DETECTOR', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                                (1, 1, 1), 2)
                                else:
                                    cv2.putText(frame, 'ID : %d' % (d[4]), (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.75,
                                                colours[d[4] % 32, :] * 255, 2)

                        if not no_display:
                            frame = cv2.resize(frame, (0, 0), fx=show_rate, fy=show_rate)
                            cv2.imshow("Frame", frame)
                            print(frame.shape)
                            videoWriter.write(frame)

                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                    except:
                        continue
            # videoWriter.release()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", type=str,
                        help='Path to the data directory containing aligned your face patches.', default='videos')
    parser.add_argument('--output_path', type=str,
                        help='Path to save face',
                        default='facepics')
    parser.add_argument('--detect_interval',
                        help='how many frames to make a detection',
                        type=int, default=1)
    parser.add_argument('--margin',
                        help='add margin for face',
                        type=int, default=10)
    parser.add_argument('--scale_rate',
                        help='Scale down or enlarge the original video img',
                        type=float, default=0.7)
    parser.add_argument('--show_rate',
                        help='Scale down or enlarge the imgs drawn by opencv',
                        type=float, default=1)
    parser.add_argument('--face_score_threshold',
                        help='The threshold of the extracted faces,range 0<x<=1',
                        type=float, default=0.85)
    parser.add_argument('--face_landmarks',
                        help='Draw five face landmarks on extracted face or not ', action="store_true")
    parser.add_argument('--no_display',
                        help='Display or not', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
