import base64
import os

import cv2
import numpy as np

import config


# ----- MsCelebV1-Faces-Aligned.tsv -----
# File format: text files, each line is an image record containing 7 columns, delimited by TAB.
# Column1: Freebase MID
# Column2: ImageSearchRank
# Column3: ImageURL
# Column4: PageURL
# Column5: FaceID
# Column6: FaceRectangle_Base64Encoded(four floats, relative coordinates of UpperLeft and BottomRight corner)
# Column7: FaceData_Base64Encoded


def get_image_record(buffer):
    image_records = list()
    line_split_data = buffer.split('\n')

    original_buffer = buffer  # only for debug

    for which_line in range(len(line_split_data)-1):
        image_record = list()
        line_data = line_split_data[which_line].split('\t')
        try:
            for which_columns in range(7):
                image_record_column = line_data[which_columns]
                image_record.append(image_record_column)
                buffer = buffer[len(image_record_column) + 1:]  # '\t' also been deleted
        except:
            continue  # data of image record is un-complete
        image_records.append(image_record)
    return buffer, image_records


def read_images(filepath, volume=1000000):
    # volumn: how many data can be read each time.
    # count = 0  # only for debug
    start_index = 0
    # image_records = list()
    if not os.path.exists(config.dataset_path):
        os.mkdir(config.dataset_path)
    with open(filepath, 'r') as fr:
        # first part of file
        fr.seek(start_index)
        buffer = fr.read(volume)
        start_index += volume
        while buffer != None:
            # get image records
            buffer, part_of_image_records = get_image_record(buffer)
            # image_records += part_of_image_records  # read all image records to memory
            # count += len(part_of_image_records)

            # save image
            try:
                for record_no in range(len(part_of_image_records)):
                    # Freebase MID as people name
                    people_name = part_of_image_records[record_no][0]
                    if not os.path.exists(os.path.join(config.dataset_path, people_name)):
                        os.mkdir(os.path.join(config.dataset_path, people_name))
                    # ImageSearchRank as image filename
                    image_filename = '%s.png' % part_of_image_records[record_no][1]
                    # change format to image
                    base64_decode = base64.b64decode(part_of_image_records[record_no][6])
                    img_data = np.fromstring(base64_decode, dtype=np.uint8)
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    print(os.path.join(os.path.join(config.dataset_path, people_name), image_filename))
                    cv2.imwrite(os.path.join(os.path.join(config.dataset_path, people_name), image_filename), img)
            except:
                None

            # next part of file
            fr.seek(start_index)
            buffer += fr.read(volume)
            start_index += volume

            # show images
            # done_count = 1000
            # if count % done_count == 0 and count != 0:
            #     print('%d images done' % done_count)
            #     cv2.imshow('image', img)
            #     cv2.waitKey()

if __name__ == '__main__':
    read_images(os.path.expanduser('/workspace/dataset/MscelebV1/MsCelebV1-Faces-Aligned.tsv'))
