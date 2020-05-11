# Semantic Segmentation Demos - Object Extraction and Background Swapping
This project includes demo programs which use the semantic segmentation deep-learning models.  
Each demo uses the DL models from OpenVINO [Open Model Zoo](https://docs.openvinotoolkit.org/latest/_models_intel_index.html) and [Intel(r) Distribution of OpenVINO(tm) toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) for deep-learning inferencing.  
 1. Object Extraction
   This demo program performs semantic segmentation to obtain the mask images and the bounding boxes of the specified class objects in an image, then uses an inpainting DL model to compensate the background of the objects. User can drag the detected objects.  
 2. Background Swapping
   This demo program performs semantic segmentation to obtain the mask image for person. The program apply and-or logical operations to swap the background with another image.   

このプロジェクトはセマンティックセグメンテーション・ディープラーニングモデルを使用したデモプログラムを含んでいます。  
どちらのデモプログラムもOpenVINO [Open Model Zoo](https://docs.openvinotoolkit.org/latest/_models_intel_index.html)のディープラーニングモデルを使用し、推論には[Intel(r) Distribution of OpenVINO(tm) toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)を使用しています。  
 1. Object Extraction
   このデモプログラムはセマンティックセグメンテーションを行い、指定したクラスのオブジェクトのマスクイメージとバウンディングボックスを取得します。その後、Inpainting (画像修復)モデルを使用してオブジェクトの背景を補間します。ユーザーはオブジェクトをドラッグして移動することが可能です。  
 2. Background Swapping
   このデモプログラムはセマンティックセグメンテーションを行い人のマスク画像を取得します。プログラムはand-or論理演算を用いて背景を別の画像に差し替えます。  

### Object Extraction Result   
![object-extracvtion](./resources/object-extraction.gif)

### Back Ground Swap Result   
![background-swap](./resources/background-swap.gif)



### Required DL Models to Run This Demo

The demo expects the following models in the Intermediate Representation (IR) format:

  * `deeplabv3`         # Semantic segmentation
  * `gmcnn-places2-tf`  # Image inpainting

You can download these models from OpenVINO [Open Model Zoo](https://github.com/opencv/open_model_zoo).
In the `models.lst` is the list of appropriate models for this demo that can be obtained via `Model downloader`.
Please see more information about `Model downloader` [here](../../../tools/downloader/README.md).

## How to Run


### 0. Prerequisites
- **OpenVINO 2020.2**
  - If you haven't installed it, go to the OpenVINO web page and follow the [*Get Started*](https://software.intel.com/en-us/openvino-toolkit/documentation/get-started) guide to do it.  

### 1. Install dependencies  
The demo depends on:
- `numpy`
- `opencv-python`

To install all the required Python modules you can use:

``` sh
(Linux) pip3 install -r requirements.in
(Win10) pip install -r requirements.in
```

### 2. Download DL models from OMZ
Use `Model Downloader` to download the required models.
``` sh
(Linux) python3 $INTEL_OPENVINO_DIR/deployment_tools/tools/model_downloader/downloader.py --list models.lst
        python3 $INTEL_OPENVINO_DIR/deployment_tools/tools/model_downloader/converter.py --list models.lst
(Win10) python "%INTEL_OPENVINO_DIR%\deployment_tools\tools\model_downloader\downloader.py" --list models.lst
        python "%INTEL_OPENVINO_DIR%\deployment_tools\tools\model_downloader\converter.py" --list models.lst
```

### 3. Run the demo app (`object-extraction.py`)
Attach a USB webCam as input of the demo program, then run the program. If you want to use a movie file as an input, you can modify the source code to do it.  

#### Operation: (object-extraction.py only)
Hit space bar to freeze the frame. The program will perform a semantic segmentation and image inpainting. This may take a couple of seconds (depends on the performance of your PC). Then, you'll see the detected objects flashing periodically. You can drag the objects with the mouse. Hit space bar again to unfreeze and continue.  

``` sh
(Linux) python3 object-extraction.py
(Win10) python object-extraction.py
```

### 3. Run the demo app (`background_swap.py`)
Attach a USB webCam as input of the demo program, then run the program. If you want to use a movie file as an input, you can modify the source code to do it.  
The program expects `background.jpg` is in the same directory.  

``` sh
(Linux) python3 background_swap.py
(Win10) python background_swap.py
```

## Demo Output  
The application draws the results on the input image.

## Tested Environment  
- Windows 10 x64 1909 and Ubuntu 18.04 LTS  
- Intel(r) Distribution of OpenVINO(tm) toolkit 2020.2  
- Python 3.6.5 x64  

## See Also  
* [Using Open Model Zoo demos](../../README.md)  
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)  
* [Model Downloader](../../../tools/downloader/README.md)  

