import argparse

from ultralytics import YOLO
import cv2
import numpy as np

import supervision as sv

from collections import deque,defaultdict

import torch


SOURCE = np.array([[1252,787],[2298,803],[5039,2159],[-550,2159]])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

width= 1280
height= 7200
TARGET=np.array([[0,0],[TARGET_WIDTH-1,0],[TARGET_WIDTH-1,TARGET_HEIGHT-1],[0,TARGET_HEIGHT-1]])

class ViewTransformer:
    def __init__(self,source:np.ndarray,target:np.ndarray):
        source=source.astype(np.float32)
        target=target.astype(np.float32)
        self.m=cv2.getPerspectiveTransform(source,target)
    def transform_points(self,points:np.ndarray)->np.ndarray:
        reshaped_points=points.reshape(-1,1,2).astype(np.float32)
        transformed_points=cv2.perspectiveTransform(reshaped_points,self.m)
        return transformed_points.reshape(-1,2)

        
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Inference and Supervision"
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    video_info=sv.VideoInfo.from_video_path(args.source_video_path)
    model= YOLO("yolov8x.pt")

    byte_tracker= sv.ByteTrack(frame_rate=video_info.fps)

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    # Get the current device, either 'cuda' or 'cpu'
    current_device = torch.cuda.current_device()
    print(f"Current device: {current_device}")

    # Get the name of the current device
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Device name: {device_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device)


    thickness= sv.calculate_dynamic_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scaler= sv.calculate_dynamic_text_scale(resolution_wh=video_info.resolution_wh)
    bounding_box_annotator= sv.BoundingBoxAnnotator(thickness=thickness)
    label_annotator= sv.LabelAnnotator(text_scale=text_scaler,text_thickness=thickness,text_position=sv.Position.BOTTOM_CENTER,color_lookup=sv.ColorLookup.TRACK)
    trace_annotator= sv.TraceAnnotator(thickness=thickness,trace_length=video_info.fps*2,position=sv.Position.BOTTOM_CENTER,color_lookup=sv.ColorLookup.TRACK)

    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    polygon_zone= sv.PolygonZone(SOURCE,frame_resolution_wh=video_info.resolution_wh)
    view_transformer=ViewTransformer(SOURCE,TARGET)
    cv2.imwrite("frame.jpg",next(frame_generator)) 
    cv2.namedWindow("Annotated Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotated Frame", width, height)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    for frame in frame_generator:
        results = model(frame)[0]
        detections= sv.Detections.from_ultralytics(results)
        detections= detections[polygon_zone.trigger(detections)]
        detections= byte_tracker.update_with_detections(detections=detections)

        points=detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points= view_transformer.transform_points(points=points).astype(int)

        labels=[]
        for tracker_id,[_,y] in zip(detections.tracker_id,points):
            coordinates[tracker_id].append(y)
            if len(coordinates[tracker_id])<video_info.fps/2:
                labels.append(f"#{tracker_id}")
            else:
                coordinate_start=coordinates[tracker_id][-1]
                coordinate_end=coordinates[tracker_id][0] 
                distance=abs(coordinate_end-coordinate_start)
                time=len(coordinates[tracker_id])/video_info.fps
                speed=distance/time*3.6
                labels.append(f"#{tracker_id} {int(speed)} km/h")



        annotated_frame=frame.copy()
        annotated_frame= trace_annotator.annotate(scene=annotated_frame,detections=detections)
        annotated_frame= bounding_box_annotator.annotate(scene=annotated_frame,detections=detections)
        annotated_frame= label_annotator.annotate(scene=annotated_frame,detections=detections,labels=labels)
        cv2.imshow("Annotated Frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()