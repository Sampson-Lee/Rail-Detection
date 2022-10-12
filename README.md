# Railroad-Detection
Rail detection, essential for railroad anomaly detection, aims to identify the railroad region in video frames. Although various studies on rail detection exist, neither an open benchmark nor a high-speed network is available in the community, making algorithm comparison and development difficult. Inspired by the growth of lane detection, we propose a rail database and a row-based rail detection method. 

# Rail-DB
We present a real-world railway dataset, Rail-DB, with 7432 pairs of images and annotations. The images are collected from different situations in lighting, road structures, and views. The rails are labeled with polylines, and the images are categorized into nine scenes. The Rail-DB is expected to facilitate the improvement of rail detection algorithms. The collection pipeline is shown in Fig.1.

<p align="center">
  <img src="./images/dataset_collection.png" />
  Fig.1 - image collection.
</p>
<!-- ![image](./images/dataset_collection.png) -->

:star: You can download the dataset by filling out this [form](https://docs.google.com/forms/d/e/1FAIpQLSemB6S2Oai4oC_mI2jxYb-KVfOVflmqY1scxEUtV24_-YP0aQ/viewform). An email with dataset download link will come to you. 



# Rail-Net
We present an efficient row-based rail detection method, Rail-Net, containing a lightweight convolutional backbone and an anchor classifier. Specifically, we formulate the process of rail detection as a row-based selecting problem. This strategy reduces the computational cost compared to alternative segmentation methods.

<p align="center">
  <img src="./images/railnet_arch.png" />
  Fig.2 - Rail-Net archetecture.
</p>

:star: train scripts

```sh
git clone git@github.com:Sampson-Lee/Rail-Detection.git
conda env create -f environment.yaml # then install your own torch
bash launch_training.sh # after specify configs/raildb.py
```

<!-- :star:other scripts

We also implement hand-crafted and segmentation methods for rail detection in this resposity. 

In train.py, we can show rail detection results by setting savefig in validate function.  -->



# Experiments
We evaluate the Rail-Net on Rail-DB with extensive experiments, including cross-scene settings and network backbones ranging from ResNet to Vision Transformers. Our method achieves promising performance in terms of both speed and accuracy. Notably, a lightweight version could achieve 92.77\% accuracy and 312 frames per second. The Rail-Net outperforms the traditional method by 50.65\% and the segmentation one by 5.86\%.

<p align="center">
  <img src="./images/results_comparison.png" />
  Fig.3 - Quantitative and qualitative results.
</p>


:star: get pretrained models from [here](https://drive.google.com/file/d/1vd8rbUEkeoHpGP4QR0dc6LrS2un2FAF3/view?usp=sharing) and deploy in real environments
```sh
cd utils
python deploy.py # after overide the image or video in this file
```

We find the pretrained model fails to generalize to many real situations (e.g., example.mp4). 😅 Therefore, we will mainly address this problem in the future work.

# Citation

Do not forget to cite our work appropriately. Rail Detection: An Efficient Row-based Network and A New Benchmark. (ACMMM 2022 Poster)
