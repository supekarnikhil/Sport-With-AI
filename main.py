## import packages
import cv2
import argparse
from utils import *
import mediapipe as mp
from types_of_exercise import TypeOfExercise

## setup agrparse
ap = argparse.ArgumentParser()
ap.add_argument("-t",
                "--exercise_type",
                type=str,
                help='Type of activity to do',
                required=True)
ap.add_argument("-vs",
                "--video_source",
                type=str,
                help='Type of activity to do',
                required=False)
args = vars(ap.parse_args())

## drawing body
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

## setting the video source
if args["video_source"] is not None:
    cap = cv2.VideoCapture(args["video_source"])
else:
    cap = cv2.VideoCapture(0)  # webcam

# cap.set(3, 800)  # width
# cap.set(4, 480)  # height

## setup mediapipe
with mp_pose.Pose(min_detection_confidence=0.7,
                  min_tracking_confidence=0.7) as pose:

    counter = 0  # movement of exercise
    status = True  # state of move
    while cap.isOpened():
        ret, frame = cap.read()
        # result_screen = np.zeros((250, 400, 3), np.uint8)

        # frame = cv2.resize(frame, (480, 800), interpolation=cv2.INTER_AREA)
        ## recolor frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        ## make detection
        results = pose.process(frame)
        ## recolor back to BGR
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            counter, status = TypeOfExercise(landmarks).calculate_exercise(
                args["exercise_type"], counter, status)
        except:
            pass

        # score_table(args["exercise_type"], counter, status)

        ## render detections (for landmarks)
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 255),
                                   thickness=2,
                                   circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 0),
                                   thickness=2,
                                   circle_radius=2),
        )
        # flipped = cv2.flip(frame, 1)
        flipped = frame

        # cv2.rectangle(flipped, (0, 20), (250, 130), (0,0,0), -1)

        cv2.putText(flipped, "Exercise : " + args["exercise_type"].replace("-", " "),
                (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2,
                cv2.LINE_AA)
        cv2.putText(flipped, "Reps : " + str(counter), (90, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', flipped)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()