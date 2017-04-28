# Tricks
1. High resolution classiﬁer: YOLO9000
2. Anchor boxes clustering: YOLO9000
3. direct location prediction: YOLO9000
4. Fine-grained features (passthrough layer): YOLO9000
5. Multi-scale training: YOLO9000
6. Dataset combination with WordTree: YOLO9000
7. Scaling of residuals: Inception-Resnet V2
8. Feature pyramid network for RPN/Fast RCNN: FPN
9. iterative regression: S. Gidaris and N. Komodakis. Object detection via a multiregion & semantic segmentationaware CNN model. ICCV, 2015.
10. hard negative mining: A. Shrivastava, A. Gupta, and R. Girshick. Training region-based object detectors with online hard example mining. CVPR, 2016.
11. context modeling: K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. CVPR, 2016.
12. stronger data augmentation: 
(1) W. Liu, D. Anguelov, D. Erhan, C. Szegedy, and S. Reed. SSD: Single shot multibox detector. ECCV, 2016.
(2) Howard, A.G.: Some improvements on deep convolutional neural network based image classiﬁcation. 2013.
(3) change illumination: Imagenet classification with deep convolutional neural networks. 2012.
13. use an image scale of 800 pixels instead of 600 in <Fast R-CNN> / <Deep residual learning for image recognition>: FPN
14. train with 512 RoIs per image which accelerate convergence, in contrast to 64 RoIs in <Fast R-CNN> / <Deep residual learning for image recognition>: FPN
15. use 5 scale anchors instead of 4 in <Deep residual learning for image recognition> (adding 32*32): FPN
16. At test time we use 1000 proposals per image instead of 300 in <Deep residual learning for image recognition>: FPN
17. ROI Align: Mask R-CNN
18. Multi-task training (add segmentation task): Mask R-CNN
19. ResNeXt-101 instead of ResNet-101: Mask R-CNN