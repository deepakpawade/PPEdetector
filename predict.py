import argparse
import sys 
sys.path.extend('./')
from predict_image import process_image
from predict_video import process_video
from predict_camera import process_camera
from predict_ip import process_ip
from ultralytics import YOLO
import logging
logging.getLogger('yolo').setLevel(logging.WARNING)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-img', type=str, default='./sample.jpg', help='image path')
    parser.add_argument('-video', type=str, default='./sample.sample.mp4', help='video path')
    parser.add_argument('-device', type=int, default=0, help='camera id')
    parser.add_argument('-urls', nargs='+', help='list of IP camera URLs')
    parser.add_argument('-model', type=str, default="./models/best_10Class_20Epochs.pt", help='model path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model = YOLO(args.model)

    if args.img is not None:
        model = YOLO(args.model)
        process_image(args.img, model)

    elif args.video is not None: 
        model = YOLO(args.model)
        process_video(args.video, model)

    elif args.urls is not None:
        model = YOLO(args.model)
        process_ip_cameras(args.urls, model)
        caps = []
        for url in args.urls:
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # set buffer size to 1
            caps.append(cap)

        # start threads for each camera
        threads = []
        for i, cap in enumerate(caps):
            t = threading.Thread(target=process_ip, args=(model, cap, i+1))
            t.start()
            threads.append(t)

        # wait for threads to finish
        for t in threads:
            t.join()

    elif args.device is not None:
        model = YOLO(args.model)
        process_camera(args.device, model)

    else:
        print("No argument was provided.")
if __name__ == '__main__':
    main()
