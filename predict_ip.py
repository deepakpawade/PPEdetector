from ultralytics import YOLO
import cv2
import argparse
import threading

class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

# multithreading method
def process_camera(model, cap, cam_id):
    while True:
        # read a frame from the video
        ret, img = cap.read()
        if not ret:
            break

        # process the frame with YOLO
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # getting confidence level
                conf = round(box.conf[0] * 100)
                # class names
                cls = class_names[int(box.cls[0])]

                text = f"{cls}: {conf}%"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
                text_offset_x = x1 + (x2 - x1) // 2 - text_width // 2
                text_offset_y = y1 - text_height

                if cls in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'] and conf > 40:
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1)
                    cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255), thickness=2)
                else:
                    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1)
                    cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=2)

        cv2.imshow(f"Camera {cam_id}", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-urls', nargs='+', help='list of IP camera URLs')
    parser.add_argument('-model', type=str, default="./models/best_10Class_20Epochs.pt", help='model path')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model = YOLO(args.model)

    # create capture objects for each camera
    caps = []
    for url in args.urls:
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # set buffer size to 1
        caps.append(cap)

    # start threads for each camera
    threads = []
    for i, cap in enumerate(caps):
        t = threading.Thread(target=process_camera, args=(model, cap, i+1))
        t.start()
        threads.append(t)

    # wait for threads to finish
    for t in threads:
        t.join()

if __name__ == '__main__':
    main()


# simple way
# from ultralytics import YOLO
# import cv2

# # initialize YOLO models for both cameras
# model1 = YOLO("./models/best_10Class_20Epochs.pt")
# model2 = YOLO("./models/best_10Class_20Epochs.pt")

# # initialize capture objects for both cameras
# cap1 = cv2.VideoCapture("http://camera1_ip_address:port/stream")
# cap2 = cv2.VideoCapture("http://camera2_ip_address:port/stream")

# # set camera properties (if needed)
# cap1.set(3, 640)
# cap1.set(4, 480)
# cap2.set(3, 640)
# cap2.set(4, 480)

# # read and process frames from both cameras
# while True:
#     # read frames from both cameras
#     success1, img1 = cap1.read()
#     success2, img2 = cap2.read()

#     # predict objects in frames from both cameras
#     if success1:
#         results1 = model1(img1, stream=True)
#         # process results from camera 1

#     if success2:
#         results2 = model2(img2, stream=True)
#         # process results from camera 2

#     # display frames (if needed)
#     cv2.imshow("Camera 1", img1)
#     cv2.imshow("Camera 2", img2)

#     # exit on key press (if needed)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # release capture objects and close windows
# cap1.release()
# cap2.release()
# cv2.destroyAllWindows()