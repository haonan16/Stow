"""
opencv renderer class.
"""
import cv2
import numpy as np


class OpenCVRenderer:
    def __init__(self, sim):
        # TODO: update this appropriately - need to get screen dimensions
        self.width = int(1280/2)
        self.height = int(800/2)

        self.sim = sim
        self.camera_name = self.sim.model.camera_id2name(0)

        self.keypress_callback = None
        
        self._num_cam = 1
        self._rendered_cam =1

    def set_camera(self, camera_id):
        """
        Set the camera view to the specified camera ID.
        Args:
            camera_id (int): id of the camera to set the current viewer to
        """
        self.camera_name = self.sim.model.camera_id2name(camera_id)

    def render(self):
        def offscreen_im():
            # get frame with offscreen renderer (assumes that the renderer already exists)
            im = self.sim.render(camera_name=self.camera_name, height=self.height, width=self.width)[..., ::-1]

            # write frame to window
            im = np.flip(im, axis=0)
            
            return im
        ims = []
        for cam_id in range(self.rendered_cam):
            self.set_camera(cam_id)
            im = offscreen_im()
            ims.append(im)
        if self.rendered_cam > 1:
            num_im_row = int(self.rendered_cam/2)
            numpy_horizontal_concat1 = np.concatenate((ims[:num_im_row]), axis=1)
            numpy_horizontal_concat2 = np.concatenate((ims[num_im_row:]), axis=1)
            numpy_vertical_concat = np.concatenate((numpy_horizontal_concat1, numpy_horizontal_concat2), axis=0)
            cv2.imshow("offscreen render", numpy_vertical_concat)
        else: 
            num_im_row = self.rendered_cam
            cv2.imshow("offscreen render", im)


        key = cv2.waitKey(1)
        if self.keypress_callback:
            self.keypress_callback(key)
    
    
    def add_keypress_callback(self, keypress_callback):
        self.keypress_callback = keypress_callback

    def close(self):
        """
        Any cleanup to close renderer.
        """

        # NOTE: assume that @sim will get cleaned up outside the renderer - just delete the reference
        self.sim = None

        # close window
        cv2.destroyAllWindows()

    @property
    def num_cam(self):
        return self._num_cam

    @num_cam.setter
    def num_cam(self, value):
        self._num_cam = value

    @property
    def rendered_cam(self):
        return self._rendered_cam

    @rendered_cam.setter
    def rendered_cam(self, value):
        assert value <= self.num_cam
        self._rendered_cam = value

