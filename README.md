# CTPN
This is a scene text detection project based on CTPN (Connectionist Text Proposal Network).You can find the paper [here](https://arxiv.org/abs/1609.03605). It's implemented in tensorflow2.3
# enviroment and libraies we used
windows 10
python3.6
tensorflow2.3.0
numpy1.18.5
opencv-python4.4.0
# demo
1.Put your images into the directory "demo_data"  
2.In windows 10, run the command line and change the working directory into the CTPN, such as "D:\PycharmProjects\CTPN"  
3.Run "python demo.py"  
4.The output files are in the directory "demo_result"
# train
## prepare training data
1. We used the data of ICDAR2017 Competition on Multi-lingual scene text detection and script identification. You can download it [here](https://rrc.cvc.uab.es/?ch=8&com=downloads). It needs to register. Or you can use your own images and ground truth labels.
2. The text area in ground truth files downloaded above are polygones.We split the polygones into target boxed(width=16),using the gt_split.py
3. We get the labels, target vetor V in vertical and target vector O in herizontal, and save the as TFRecords files using the gen_tfrecords.py
## training loop
We define the CTPN model and train it in the model_train.py file.
Note: We used the tf.function to accelerate. If you want to debug, uncomment the line 14 "#tf.config.run_functions_eagerly(True)". If you do that, the tensorflow run in eagor mode. You can debug the whole script,but it runs slowly.
