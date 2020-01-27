import cv2
from PIL import Image
from yolo import YOLO
import time
from resizevideo import take_and_resize
import numpy as np



def gstreamer_pipeline(
    capture_width=314,
    capture_height=314,
    display_width=314,
    display_height=314,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def show_camera():
	yolo = YOLO('model_data/yolo-tiny.h5', 'model_data/tiny_yolo_anchors.txt', 'model_data/coco_classes.txt')


    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)


    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, frame = cap.read()
            frame = take_and_resize(frame)
            outBoxes = yolo.detect_image(frame)
			frame = np.asarray(frame)
			if len(outBoxes) > 0:
				for box in outBoxes:
					# extract the bounding box coordinates
					(x, y) = (int(box[0]), int(box[1]))
					(w, h) = (int(box[2]), int(box[3]))
					bbox = [x, y, w, h, box[4]]
					output_classes.append(bbox[4])
					cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
					text = 'classID = {}'.format(box[4])
					cv2.putText(frame, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

			cv2.imshow("CSI Camera", img)
            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
		yolo.close_session()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    show_camera()
