# Real Time Object Detection System
One of the most important and challenging task in computer vision is **Object Detection**. Object detection is the process of identifying instances of real-world objects, such as cars, faces, bicycles, etc. in images or videos. This process has profound use in different applications like video surveillance, face detection, image retrieval, advanced driver assistance systems (ADAS), etc.  Fast and accurate object detection models would enable autonomous driving without the need of any specialized sensors, improve human-computer interaction by use of intelligent devices to convey real-time information to human users, aid in building next-generation fast responsive robots, etc. 

Our project aims to perform *object detection* of a real time video stream. We plan to analyze existing state-of-the-art architectures in this field and propose a network for the said task. We investigate the use of single network to predict bounding boxes and and class probabilities directly from full images. Initially, we propose to test our network on images from different labeled datasets and later proceed to test the same on real time video stream. Finally, we propose to optimize the network to achieve optimum speed-accuracy of object detection and compare with existing architectures.

## Team members
- Soumendu Kumar Ghosh [GitHubUserA](https://github.com/soumendukrg)
- Arindam Bhanja Chowdhury [GitHubUserB](https://github.com/abhanjac)

## Goals
* Analyze different available deep neural network architectures and propose a network architecture to perform object detection on images from the VOC2012 dataset.
* Perfom real time object detection in a live video stream.
* Analyze accuracy vs speed of detection and optimize the network for real time detection.
* Compare proposed network performance with that of existing architectures.

## Challenges
* Prior work in this field repurposes classifiers to carry out detection. These existing systems uses a classifier for the object it aims to detect and perform evaluation at various locations and scales in a test image. For example, methods like R-CNN [1] generates potential bounding boxes in an image, classifies these boxes, and finally removes duplicate boxes to predict the class and the bounding box. Some other systems use sliding window for detection. However, such complex networks, though highly accurate, are very slow and not ideal for detection in real time. YOLO [2] is a state-of-the-art, real-time object detection system which uses single network to perform both classification and detection in a single evaluation. After initial analysis, we plan on using the YOLO architecture as the base reference model and propose to optimize the performance of the architecture by making any necessary modifications.

## References
[1] Girshick, Ross, et al. "Rich feature hierarchies for accurate object detection and semantic segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014.

[2] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.
