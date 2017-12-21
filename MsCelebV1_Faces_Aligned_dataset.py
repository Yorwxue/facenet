import base64

import numpy as np
from scipy import misc
import cv2

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
    image_records = list()
    with open(filepath, 'r') as fr:
        # first part of file
        fr.seek(start_index)
        buffer = fr.read(volume)
        start_index += volume
        while buffer != None:
            # get image records
            buffer, part_of_image_records = get_image_record(buffer)
            image_records += part_of_image_records
            # count += len(part_of_image_records)

            # next part of file
            fr.seek(start_index)
            buffer += fr.read(volume)
            start_index += volume

            # show images
            # done_count = 1000
            # if count % done_count == 0 and count != 0:
            #     print('%d images done' % done_count)
            #     # change format to image
            #     base64_decode = base64.b64decode(image_records[-1][6])
            #     img_data = np.fromstring(base64_decode, dtype=np.uint8)
            #     img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            #     cv2.imshow('image', img)
            #     cv2.waitKey()

if __name__ == '__main__':
    read_images('/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset/MscelebV1/MscelebV1-Faces-Aligned/MsCelebV1-Faces-Aligned.tsv')
