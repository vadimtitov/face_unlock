import os
import time
import sys

import cv2
import numpy as np
import face_recognition
from pynput.keyboard import Key, Controller

DIR = "face_unlock/faces/"
authorized_names = [name.split(".")[0] for name in os.listdir(DIR)]
keyboard = Controller()


def screen_is_locked():
    with os.popen("loginctl show-user | grep IdleHint") as process:
        if "yes" in process.read():
            return True
        else:
            return False


def do_when_recognized():
    """Lines to run when face is recognized.
    """
    # unlock the screen
    os.system("loginctl unlock-session")

    # wake the screen up
    keyboard.press(Key.ctrl)
    keyboard.release(Key.ctrl)


class RecognitionSequence:
    """Stores a fixed length sequence of "face names" found in
    each video frame and determines whether appearance of
    one face is frequent enough to be considered as recognized.
    """

    def __init__(self, length, required_prob):
        """Init."""
        self.length = length
        self.required_prob = required_prob
        self.sequence = []

    def __str__(self):
        return str(self.sequence)

    def append(self, name):
        self.sequence.append(name)
        if len(self.sequence) > self.length:
            del self.sequence[0]

    @property
    def current_prob(self):
        """Calculate current rate of appearance of an authorized "face name"."""
        return sum(
            1 if name in authorized_names else 0 for name in self.sequence
        )/self.length

    def recognized(self):
        """Determine whether the sequence contains enough
        appearances of an authorized face.
        """
        if self.current_prob >= self.required_prob:
            self.sequence = []
            return True
        return False


def run_face_unlock(cam_source=0, do_when_recognized=do_when_recognized):
    """This code is partialy taken from face_recognition/examples repository."""
    video_capture = cv2.VideoCapture(cam_source)

    known_face_names = []
    known_face_encodings = []

    for file in os.listdir(DIR):
        name = file.split(".")[0].split("_")[0]
        known_face_names.append(name)

        image = face_recognition.load_image_file(DIR + file)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)

    face_locations = []
    face_encodings = []

    # In any 7 consequent frames we want a face to be found at least 5 times
    rec_sequence = RecognitionSequence(
        length=7,
        required_prob=5/7
    )

    # Process every second frame for better performance
    process_this_frame = True
    while True:

        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Don't process if screen is unlocked
        if not screen_is_locked():
            time.sleep(5)
            continue

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                rec_sequence.append(name)

                if rec_sequence.recognized():
                    print("Face recognized:", name)
                    do_when_recognized()

        process_this_frame = not process_this_frame

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


def main():
    try:
        cam_source = int(sys.argv[1])
    except IndexError:
        cam_source = 0

    run_face_unlock(cam_source, do_when_recognized)


if __name__ == '__main__':
    main()
