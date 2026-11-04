import cv2

DICT = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_APRILTAG_36h11
)

detector = cv2.aruco.ArucoDetector(DICT)

cameras = [(2, "Camera 1"), (0, "Camera 2")]

for index, name in cameras:
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"{name}: Cannot open /dev/video{index}")
        continue

    print(f"\n{name}: Press S to detect, Q to continue.")

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        corners, ids, rejected = detector.detectMarkers(frame)

        display = frame.copy()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(
                display, corners, ids
            )

            detected = ids.flatten().tolist()

            cv2.putText(
                display,
                f"Detected IDs: {detected}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow(name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if ids is not None:
                print("Detected AprilTag IDs:", ids.flatten().tolist())
            else:
                print("No AprilTags detected")

        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
