import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ROIReflection:
    def __init__(self, image, hyperimage):
        self.image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.image = self.image.astype(np.uint8)
        self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        self.hyperspectral = hyperimage
        self.right_clicks = []        

    def mouse_callback(self, event, x, y, flags, params):
        #right-click event value is 2
        if event == cv2.EVENT_RBUTTONDOWN:        
            #store the coordinates of the right-click event
            self.right_clicks.append([x, y])
        

    def select_regions(self):
        cv2.namedWindow("Select the regions", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Select the regions', self.mouse_callback)
        cv2.imshow('Select the regions', self.image)

        # Wait for the user to make selections and close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Return the list of right-click coordinates
        return self.right_clicks
    
    def extract_roi(self, area, x, y, w, h):
        roi = area[:, y:y+h, x:x+w]
        
        return roi
    
    def get_roi(self, length=1):
        # Retrieves right-click coordinates and processes the hyperspectral image
        coordinates = self.right_clicks
        rois = []  # Store extracted ROIs
        length = length
        image_input = self.hyperspectral.copy()

        for coordinate in coordinates:
            x, y = coordinate
            x1 = y
            y1 = self.image.shape[1] - x
            roi = self.extract_roi(image_input, x1, y1, length, length)
            rois.append(roi)
        
        return rois

    def av_reflection(self, length=1):
        # Calculate mean intensity for each band
        reflectance = []
        rois = self.get_roi(length)
        for roi in rois:
            intensity = [np.mean(roi[b, :, :]) for b in range(roi.shape[0])]
            reflectance.append(intensity)

        # Average intensity over all regions of interest
        #avg_reflectance = np.mean(reflectance, axis=0) if reflectance else None

        return reflectance
    
    def visual_roi(self, width=1, linewidth=1, color='white'):
        # Create a plot to show the image
        fig, ax = plt.subplots(figsize=(8, 18))
        ax.imshow(self.image, cmap='gray')
        ax.set_title("RGB")

        # Add rectangles for each ROI
        for x, y in self.right_clicks:
            rect = patches.Rectangle(
                (x - width, y - width),  # Bottom-left corner
                2 * width,  # Width of the rectangle
                2 * width,  # Height of the rectangle
                linewidth=linewidth,
                edgecolor=color,
                facecolor='none'  # Transparent inside
            )
            ax.add_patch(rect)

        plt.show()
