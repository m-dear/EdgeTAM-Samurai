import os
import cv2
import numpy as np
import torch
import time
import urllib.request
import traceback
from edge_sam2.build_sam import build_sam2_object_tracker

torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
# turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


class SAM2Tracker:
    """
    A class to track a single object in a video using the EdgeSAM/SAM2 model.
    The user selects the initial object with a bounding box.
    """
    
    def __init__(self, checkpoint_path: str, model_cfg: str, device: str):
        """
        Initializes the tracker.

        Args:
            checkpoint_path (str): Path to the SAM2 model checkpoint.
            model_cfg (str): Path to the model configuration YAML file.
            device (str): The device to run the model on (e.g., 'cuda:0' or 'cpu').
        """
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load the SAM2 object tracker model
        self.predictor = build_sam2_object_tracker(
            config_file=self.model_cfg,
            ckpt_path=self.checkpoint_path,
            device=self.device,
            # This implementation is for a single object
            num_objects=1, 
            verbose=False
        )
        print("SAM2 predictor loaded successfully.")
    
    def select_bbox_with_mouse(self, frame: np.ndarray) -> np.ndarray:
        """
        Allows the user to select a bounding box using the mouse.

        Args:
            frame (np.ndarray): The frame to select the bounding box from.

        Returns:
            np.ndarray: The selected bounding box in [[x1, y1], [x2, y2]] format.
        """
        print("Select the object to track. Press SPACE or ENTER to confirm, ESC to cancel.")
        bbox = cv2.selectROI("Select Object", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select Object")
        
        if bbox[2] > 0 and bbox[3] > 0:  # Check for a valid selection
            x, y, w, h = bbox
            # Convert (x, y, w, h) to [[x1, y1], [x2, y2]] format for the model
            return np.array([[x, y], [x + w, y + h]], dtype=np.float32)
        else:
            raise ValueError("No valid bounding box was selected.")
    
    def create_visualization(self, frame: np.ndarray, mask: np.ndarray, frame_idx: int, fps: float = 0.0, avg_fps: float = 0.0) -> np.ndarray:
        """
        Creates a visualization by overlaying the mask on the frame and adding text info.

        Args:
            frame (np.ndarray): The original video frame.
            mask (np.ndarray): The prediction mask for the object.
            frame_idx (int): The current frame index.
            fps (float): The current rolling average FPS.
            avg_fps (float): The average FPS over the entire video.

        Returns:
            np.ndarray: The visualized frame.
        """
        # Create a copy of the frame
        result = frame.copy()
        
        # Deep Sky Blue color for overlay (#00bfff -> RGB: 0, 191, 255 -> BGR: 255, 191, 0)
        deep_sky_blue = [255, 100, 0]
        
        # Blue color for bounding box (BGR: 255, 0, 0)
        blue = [255, 0, 0]
        
        if mask.any():
            # Find contours of the mask``
            contours, _ = cv2.findContours((mask).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Similar to draw_masked approach
            mask_img = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask_img, contours, 255)
            
            # Apply colored overlay
            overlay = frame.copy()
            overlay[mask_img > 0] = deep_sky_blue
            result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
            
            # Calculate bounding box from mask and draw it in blue
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                x1, y1 = np.min(x_indices), np.min(y_indices)
                x2, y2 = np.max(x_indices), np.max(y_indices)
                cv2.rectangle(result, (x1, y1), (x2, y2), blue, 2)
        
        # Create semi-transparent background for text
        overlay_rect = result.copy()
        cv2.rectangle(overlay_rect, (10, 10), (200, 140), (0, 0, 0), -1)
        result = cv2.addWeighted(overlay_rect, 0.6, result, 0.4, 0)
        
        font_size = 0.8
        
        # Display frame number
        cv2.putText(result, f"Frame: {frame_idx}", (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 3)
        cv2.putText(result, f"Frame: {frame_idx}", (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
        
        # Current FPS with color coding based on performance
        fps_color = (0, 255, 0) if fps > 20 else (0, 165, 255) if fps > 10 else (0, 0, 255)
        cv2.putText(result, f"FPS: {fps:.1f}", (20, 80), 
            cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 3)
        cv2.putText(result, f"FPS: {fps:.1f}", (20, 80), 
            cv2.FONT_HERSHEY_SIMPLEX, font_size, fps_color, 2)
        
        # Average FPS in a different style
        cv2.putText(result, f"Avg: {avg_fps:.1f}", (20, 120), 
            cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 3)
        cv2.putText(result, f"Avg: {avg_fps:.1f}", (20, 120), 
            cv2.FONT_HERSHEY_SIMPLEX, font_size, (100, 255, 255), 2)
        
        return result
    
    def track_video(self, video_path: str, output_path: str = None):
        """
        Main tracking function to process a video file.

        Args:
            video_path (str): Path to the input video file.
            output_path (str): Path to save the output video. If None, no video is saved.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # Get video properties for output recording
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path is provided
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Recording output video to: {output_path}")
        
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read the first frame from the video.")
        
        # Select bounding box with mouse
        bbox = self.select_bbox_with_mouse(first_frame)
        print(f"Selected bbox: {bbox.tolist()}")
        
        # Reset video to the beginning to start tracking from frame 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # --- Tracking and Visualization Loop ---
        frame_count = 0
        is_first_frame = True
        print("Tracking started. Press 'q' to quit.")
        

        fps_window = 30  # Calculate FPS over the last 30 frames for a rolling average
        frame_times = []
        total_start_time = time.time()

        # Use inference_mode and autocast for performance
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16):
            while cap.isOpened():
                frame_start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert frame from BGR (OpenCV) to RGB (model)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Track object
                if is_first_frame:
                    # Provide the initial bounding box to track a new object
                    self.predictor.reset_tracker_state()  # Reset the predictor for a new object
                    sam_out = self.predictor.track_new_object(img=img_rgb, box=bbox)
                    is_first_frame = False
                else:
                    # Track all previously identified objects in subsequent frames
                    sam_out = self.predictor.track_all_objects(img=img_rgb)
                
                # Extract the mask. Assuming a single object is tracked.
                if sam_out['pred_masks'].shape[0] > 0:
                    # The mask is on the GPU and might be smaller than the frame.
                    # Move to CPU, resize it, and convert to a boolean array.
                    mask_small = (sam_out['pred_masks'][0, 0] > 0).cpu().numpy()
                    original_h, original_w = frame.shape[:2]
                    mask = cv2.resize(mask_small.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    # If tracking is lost, use an empty mask of the correct size
                    mask = np.zeros(frame.shape[:2], dtype=bool)
                
                # --- FPS Calculation ---
                frame_end_time = time.time()
                frame_duration = frame_end_time - frame_start_time
                frame_times.append(frame_duration)
                
                if len(frame_times) > fps_window:
                    frame_times.pop(0)
                
                current_fps = len(frame_times) / sum(frame_times) if sum(frame_times) > 0 else 0
                avg_fps = (frame_count + 1) / (time.time() - total_start_time) if frame_count > 0 else 0
                
                # Create and display visualization
                display_frame = self.create_visualization(frame, mask, frame_count, current_fps, avg_fps)
                cv2.imshow("SAM2 Tracking", display_frame)
                
                # Write frame to output video if recording
                if video_writer is not None:
                    video_writer.write(display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                frame_count += 1
        
        # --- Cleanup and Final Stats ---
        total_time = time.time() - total_start_time
        overall_fps = frame_count / total_time if total_time > 0 else 0
        
        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"Output video saved to: {output_path}")
        cv2.destroyAllWindows()
        print("\n--- Tracking Completed ---")
        print(f"Processed {frame_count} frames.")
        print(f"Overall Average FPS: {overall_fps:.2f}")
        print(f"Total time: {total_time:.2f} seconds")



def main():
    """Main function to configure and run the tracker."""
    # --- Configuration ---
    CONFIG = {
        "checkpoint_path": "checkpoints/edgetam.pt",
        "model_cfg": "edgetam.yaml",
        "video_url": "https://motchallenge.net/sequenceVideos/TUD-Stadtmitte-raw.webm",
        "video_path": "./TUD-Stadtmitte-raw.webm",
        "output_video_path": "./samurai_tracking_output.mp4",  # Output video path
        "device": "cuda:0"
    }

    if not os.path.exists(CONFIG["video_path"]):
        print(f"Downloading video from {CONFIG['video_url']}...")
        try:
            urllib.request.urlretrieve(CONFIG['video_url'], CONFIG['video_path'])
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading video: {e}")
            return
    else:
        print(f"Video file already exists at {CONFIG['video_path']}.")

    # --- Run Tracker ---
    try:
        tracker = SAM2Tracker(
            checkpoint_path=CONFIG["checkpoint_path"],
            model_cfg=CONFIG["model_cfg"],
            device=CONFIG["device"]
        )
        tracker.track_video(
            video_path=CONFIG["video_path"],
            output_path=CONFIG["output_video_path"]
        )
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
