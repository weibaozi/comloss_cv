# comloss_cv


Set the correct image topic from your endoscope in the main.py file, use raw image topic because all training data was taken in raw image topic.
By default, the image topic is set to: "**/jhu_daVinci/left/image_raw**"
```python
image_topic = "/jhu_daVinci/left/image_raw"
```
- **To run the code**
```bash
python3 main.py --model.pth
```


- **To train the model**
```bash
python3 mobilenetv2.py
```

- **To test the model**
```bash
python3 test.py
```

- If you facing issue the accuracy drops for current camera angle, you can take images from the endoscope and add to the dataset by:
  - put image holding the peg in the **img2/train/1** folder  
  - put image without holding the peg in the **img2/train/0** folder
  - Then run the training code again
- **libraries:**
  - numpy
  - pytorch >= 2.4
  - torchvision
  - pytorch-lightning
  - openCV