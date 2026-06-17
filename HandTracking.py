import cv2 as cv
import mediapipe as mp
from typing import List, Dict, Any


class HandTracker:

    def __init__(
        self,
        static_mode: bool = False,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7
    ) -> None:

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        self.results = None

    def detect_hands(self, frame, draw: bool = True):
      
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)

        if self.results and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:

                if draw:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )

        return frame

    def get_landmarks(
        self,
        frame,
        hand_index: int = 0
    ) -> List[Dict[str, Any]]:

        landmark_list = []

        if (
            not self.results or
            not self.results.multi_hand_landmarks or
            hand_index >= len(self.results.multi_hand_landmarks)
        ):
            return landmark_list

        selected_hand = self.results.multi_hand_landmarks[hand_index]

        height, width, _ = frame.shape

        for landmark_id, landmark in enumerate(selected_hand.landmark):

            landmark_list.append({
                "id": landmark_id,
                "x": int(landmark.x * width),
                "y": int(landmark.y * height),
                "z": landmark.z
            })

        return landmark_list

    def get_hand_label(self, hand_index: int = 0) -> str:
        
        if (
            not self.results or
            not self.results.multi_handedness or
            hand_index >= len(self.results.multi_handedness)
        ):
            return "Unknown"

        return (
            self.results
            .multi_handedness[hand_index]
            .classification[0]
            .label
        )