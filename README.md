# A naive classifier for detecting face mask
As a demo to show how to finetune a CV task.
  
The UI is shown below:

<p align="center">
  <img src="rs/demo.gif" alt="UI demo" />
</p>

<br/>
<br/>
<br/>
  
You can run the following script to start the Gradio web UI:
```bash
python3 main.py
```
  
<br/>
A brief description video is below:
<br/>
https://www.bilibili.com/video/BV18psgeNEqF
<br/>
<br/>
解释文档在<a href="rs/document.pdf">这里</a>。
<br/>

---

If you can't run it directly, you may need to do some preparation, including but not limilited to:

- Install libaray:
```bash
!pip install -U -qq torch torchvision umap-learn timm thop tensorboard dlib face_recognition
```

