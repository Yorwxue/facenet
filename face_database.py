import os

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
import json

import config
import facenet
from align import detect_face

args_margin = config.args_margin
args_image_size = config.args_image_size

minsize = config.minsize  # minimum size of face
threshold = config.threshold  # three steps's threshold
factor = config.factor  # scale factor
face_threshold = config.face_threshold

db_name = config.db_name
known_img_path = config.known_img_path
update = config.update

# register
reg_person_names = ''
reg_locations = [0, 0, 0, 0]
reg_distances = 9999
reg_not_fount_count = 0


def face_detection(image_filename):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

        # ---------------
        args_seed = 666
        # input a img of face closeup
        np.random.seed(seed=args_seed)
        src_path, _ = os.path.split(os.path.realpath(__file__))
        args_model = os.path.expanduser(src_path + '/pre_train_models/20170512-110547.pb')
        # Load the model
        print('Loading feature extraction model')
        facenet.load_model(args_model)
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        # embedding_size = embeddings.get_shape()[1]

        # ---------------
        face_closeups = list()
        face_source = list()
        face_locations = list()

        img = facenet.img_read(image_filename)

        cv2.imshow("image", img)
        cv2.waitKey(500)

        bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]

        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            img_size = np.asarray(img.shape)[0:2]

            for det_no in range(nrof_faces):
                each_det = np.squeeze(det[det_no])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(each_det[0] - args_margin / 2, 0)  # left Bound
                bb[1] = np.maximum(each_det[1] - args_margin / 2, 0)  # upper Bound
                bb[2] = np.minimum(each_det[2] + args_margin / 2, img_size[1])  # right Bound
                bb[3] = np.minimum(each_det[3] + args_margin / 2, img_size[0])  # lower Bound
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                scaled = misc.imresize(cropped, (args_image_size, args_image_size), interp='bilinear')

                # cv2.imshow("image", cropped)
                # cv2.waitKey(1000)

                face_closeups.append(scaled)
                face_source.append(image_filename)
                face_locations.append(bb)

            face_vectors, face_source, _ = facenet.faceDB(db_name, img_path=known_img_path, update=update)

            query_face_closeup = face_closeups
            query_face_source = image_filename
            query_face_locations = face_locations

            query_processed_face_ = facenet.face_process(query_face_closeup, False, False, args_image_size)
            # query_face_vector = facenet.get_face_vec(query_processed_face_)

            print('Calculating features for images')
            feed_dict = {images_placeholder: query_processed_face_, phase_train_placeholder: False}
            emb_array = sess.run(embeddings, feed_dict=feed_dict)
            query_face_vector = emb_array

            # find the people who have those faces
            # query_face_closeup, query_face_source, query_face_locations = get_face_img(np.atleast_1d(query_img_path))
            source_list = list()
            location_list = list()
            name_list = list()
            distance_list = list()

            for face_no in range(len(query_face_vector)):
                dist = facenet.cal_euclidean(query_face_vector[face_no], face_vectors)
                # indices = dist.argsort()[:3]  # find the indices of the 3 lower number
                index = dist.argsort()[:1]  # the most similar

                # threshold checking
                distance = dist[index]
                if distance > face_threshold:
                    person_name = 'unknow'
                else:
                    person_name = str(face_source[index]).split('/')[-2]
                # faceInfo.append([query_face_source[face_no], query_face_locations[face_no], person_name, distance])
                source_list.append(query_face_source[face_no])
                location_list.append(query_face_locations[face_no])
                name_list.append(person_name)
                distance_list.append(distance)

            # -------- register ---------
            # global reg_not_fount_count
            # global reg_person_names
            # global reg_locations
            # global reg_distances
            #
            # if nrof_faces == 0:
            #     if reg_not_fount_count < 10:
            #         name_list = reg_person_names
            #         face_locations = reg_locations
            #         distance_list = reg_distances
            #         reg_not_fount_count += 1
            #     else:
            #         reg_person_names = ''
            #         reg_locations = [0, 0, 0, 0]
            #         reg_distances = 9999
            # else:
            #     reg_person_names = name_list.copy()
            #     reg_locations = face_locations.copy()
            #     reg_distances = distance_list.copy()
            #     reg_not_fount_count = 0
            # ----------------------------

            person_names = name_list
            locations = face_locations
            distances = distance_list

            marked_frame = facenet.drawBoundaryBox([img] * len(emb_array), face_locations, name_list, distance_list)

            for point in range(len(points)//2):  # five points of each face
                for face_no in range(len(points[point])):
                    cv2.circle(marked_frame[0], (points[point][face_no], points[point+5][face_no]), 2, (255, 255, 255), -1)
                # cv2.circle(marked_frame[0], (447, 63), 5, (0, 0, 255), -1)

            cv2.imshow("image", marked_frame[0])
            cv2.waitKey(10000)

# ------------------------------------------------
if __name__ == "__main__":
    # demo
    image_filename = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset/face_database/林高洲/007.png'
    image_filename = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset/26543368_10215276246329042_830858535_o.jpg'
    img = facenet.img_read(image_filename)
    # face_detection(image_filename)

    # -------------------
    # update database
    db_face_vectors, db_face_source, _ = facenet.faceDB(db_name, img_path=known_img_path, update=True)
    exit()
    # -------------------
    # show database

    # output_json = list()
    #
    # for each_source in range(len(db_face_source)):
    #     name = db_face_source[each_source].split('/')[-2]
    #     vector = db_face_vectors[each_source]
    #     output_json.append({'face_name': name, "face_feature": vector})
    #
    # # with open('JSON_DB', 'w') as fw:
    # #     json.dump(output_json, fw)
    # #
    # # with open('JSON_DB', 'r') as fr:
    # #     output_json = json.load(fr)
    #
    # for each in output_json:
    #     print(each)

    # --------------------
    # get faces location
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

        args_seed = 666
        # input a img of face closeup
        np.random.seed(seed=args_seed)
        src_path, _ = os.path.split(os.path.realpath(__file__))
        args_model = os.path.expanduser(src_path + '/pre_train_models/20170512-110547.pb')
        # Load the model
        print('Loading feature extraction model')
        facenet.load_model(args_model)
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]

        # get face images
        face_closeups = list()
        face_source = list()
        face_locations = list()
        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            img_size = np.asarray(img.shape)[0:2]

            for det_no in range(nrof_faces):
                each_det = np.squeeze(det[det_no])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(each_det[0] - args_margin / 2, 0)  # left Bound
                bb[1] = np.maximum(each_det[1] - args_margin / 2, 0)  # upper Bound
                bb[2] = np.minimum(each_det[2] + args_margin / 2, img_size[1])  # right Bound
                bb[3] = np.minimum(each_det[3] + args_margin / 2, img_size[0])  # lower Bound
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                scaled = misc.imresize(cropped, (args_image_size, args_image_size), interp='bilinear')

                face_closeups.append(scaled)
                face_source.append(image_filename)
                face_locations.append(bb)
        # --------------------
        # pre-processing
        query_face_closeup = face_closeups
        query_face_source = image_filename
        query_face_locations = face_locations

        query_processed_face_ = facenet.face_process(query_face_closeup, False, False, args_image_size)

        # --------------------
        # face vectors
        feed_dict = {images_placeholder: query_processed_face_, phase_train_placeholder: False}
        emb_array = sess.run(embeddings, feed_dict=feed_dict)
        query_face_vector = emb_array

        # --------------------
        # classifier
        source_list = list()
        location_list = list()
        name_list = list()
        distance_list = list()

        for face_no in range(len(query_face_vector)):
            dist = facenet.cal_euclidean(query_face_vector[face_no], db_face_vectors)
            # indices = dist.argsort()[:3]  # find the indices of the 3 lower number
            index = dist.argsort()[:1]  # the most similar

            # threshold checking
            distance = dist[index][0]
            if distance > face_threshold:
                person_name = config.unknown
            else:
                person_name = str(db_face_source[index]).split('/')[-2]
            # faceInfo.append([query_face_source[face_no], query_face_locations[face_no], person_name, distance])
            source_list.append(query_face_source[face_no])
            location_list.append(query_face_locations[face_no])
            name_list.append(person_name)
            distance_list.append(distance)

            # print(source_list)
            print(person_name)
            # show face closeup
            for face in query_face_closeup:
                cv2.imshow("image", face)
                cv2.waitKey(1000)
            print('--------------------------')
