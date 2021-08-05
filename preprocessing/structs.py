class Point:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class RealSenseBagMetaData:
    def __init__(self, file_path: str, first_frame_number: int, last_frame_number: int,
                 number_of_padding_frames: int, total_frames_number_after_padding: int):
        self.total_frames_number_after_padding = total_frames_number_after_padding
        self.number_of_padding_frames = number_of_padding_frames
        self.last_frame_number = last_frame_number
        self.first_frame_number = first_frame_number
        self.file_path = file_path
        self.width = 640
        self.height = 480
        self.FPS = 30
        
