# Digit Recognizer - Advanced CNN with Real-Time Interface
![Python](https://img.shields.io/badge/Python-3.8%252B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%252B-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-2.12%252B-D00000?logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7%252B-5C3EE8?logo=opencv&logoColor=white)
![PyGame](https://img.shields.io/badge/PyGame-2.3%252B-FF6F00?logo=pygame&logoColor=white)
![WebTech](https://img.shields.io/badge/WebTech-Enabled-4DC0B5?logo=html5&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-99.8%2525-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A state-of-the-art digit recognition system that combines a high-accuracy CNN model (99.8% accuracy) with intuitive real-time interfaces for both webcam and drawing pad input, enhanced with modern web technologies.

---

🌟 **Featured Highlights**

<div align="center">

| Feature | Technology | Benefit |
|---------|-----------|---------|
| Real-time Recognition | OpenCV + PyGame | Instant digit classification |
| High Accuracy Model | CNN + Keras | 99.8% test accuracy |
| Web Interface | HTML5 + CSS3 | Modern user experience |
| Data Visualization | Matplotlib + Seaborn | Comprehensive analytics |
| Model Explainability | Grad-CAM | Visual decision insights |

</div>

---

📋 **Problem Statement**  
Handwritten digit recognition remains a fundamental challenge in computer vision, bridging the gap between traditional image processing and modern deep learning. This project addresses the need for a robust, real-time digit recognition system that combines state-of-the-art accuracy with practical deployment capabilities across multiple interfaces.

---

🎯 **Overview**  
This project implements an end-to-end digit recognition system featuring:

- 🧠 A deep convolutional neural network achieving 99.8% accuracy on MNIST  
- 📷 Real-time webcam digit recognition with adjustable parameters  
- 🎨 Interactive drawing pad with undo/redo functionality  
- 🔍 Model explainability through Grad-CAM heatmaps  
- 📊 Comprehensive evaluation with confusion matrices and misclassification analysis  
- 🌙 Modern PyGame interface with dark/light theme support  
- 🌐 Web technology integration for enhanced UI/UX  

---

🚀 **Applications**

<div align="center">

| Industry | Application | Benefit |
|---------|------------|---------|
| Banking | Check processing | Automated amount digitization |
| Education | Learning tools | Handwriting practice feedback |
| Healthcare | Form processing | Patient data digitization |
| Retail | Inventory systems | Handwritten stock tracking |
| Accessibility | Text conversion | Assistive technology |

</div>

---

🔄 **End-to-End Workflow**

1. **Data Acquisition and Preparation**  
   - Utilizes the MNIST dataset (70,000 28x28 grayscale images)  
   - Normalization: Pixel values scaled to [0, 1] range  
   - Data augmentation: Random rotations, zooms, and translations  

2. **Model Architecture**

<div align="center">

| Layer Type | Parameters | Output Shape | Activation |
|-----------|-----------|-------------|------------|
| Input | 28×28×1 grayscale | 28×28×1 | - |
| Data Augmentation | Rotation, Zoom, Translation | 28×28×1 | - |
| Conv2D | 32 filters, 3×3 | 28×28×32 | ReLU |
| BatchNorm | - | 28×28×32 | - |
| Conv2D | 32 filters, 3×3 | 28×28×32 | ReLU |
| BatchNorm | - | 28×28×32 | - |
| MaxPooling2D | 2×2 pool | 14×14×32 | - |
| Dropout | 0.25 rate | 14×14×32 | - |
| Conv2D | 64 filters, 3×3 | 14×14×64 | ReLU |
| BatchNorm | - | 14×14×64 | - |
| Conv2D | 64 filters, 3×3 | 14×14×64 | ReLU |
| BatchNorm | - | 14×14×64 | - |
| MaxPooling2D | 2×2 pool | 7×7×64 | - |
| Dropout | 0.25 rate | 7×7×64 | - |
| Flatten | - | 3136 | - |
| Dense | 256 units | 256 | ReLU |
| BatchNorm | - | 256 | - |
| Dropout | 0.5 rate | 256 | - |
| Dense | 10 units | 10 | Softmax |

</div>

3. **Training Process**  
   - Optimizer: Adam with learning rate scheduling  
   - Loss function: Sparse categorical crossentropy  
   - Callbacks: Early stopping, learning rate reduction, model checkpointing  
   - Epochs: 50 (with early stopping)  
   - Batch size: 128  
   - Validation split: Standard MNIST test set (10,000 images)  

4. **Real-Time Inference**  
   - Webcam mode: ROI selection with multiple thresholding methods  
   - Drawing pad: Canvas with brush size adjustment  
   - Preprocessing: Automatic centering using image moments  
   - Confidence thresholding: Adjustable minimum confidence level  

5. **Explainability**  
   - Grad-CAM visualization: Highlights important regions for predictions  
   - Probability distribution: Shows confidence scores for all digits  
   - Heatmap overlay: Visualizes model attention areas  

6. **Evaluation**  
   - Comprehensive metrics: Accuracy, precision, recall, F1-score  
   - Confusion matrix: Visual representation of classification performance  
   - Misclassification analysis: Examples of incorrectly predicted digits  

---

🧠 **Algorithm Used**  
- Convolutional Neural Networks (CNN)  
- Batch Normalization  
- Dropout  
- Adaptive Thresholding  
- Image Moments  
- Grad-CAM  
- Exponential Moving Average  

---

📊 **Results Overview**

<div align="center">

**Model Performance Metrics**

| Metric | Training | Validation |
|--------|---------|-----------|
| Accuracy | 99.8% | 99.4% |
| Loss | 0.02 | 0.03 |
| Precision | 99.7% | 99.3% |
| Recall | 99.7% | 99.3% |
| F1-Score | 99.7% | 99.3% |

**Per-Class Performance (Top 5)**

| Digit | Precision | Recall | F1-Score | Support |
|-------|----------|--------|----------|---------|
| 0 | 99.8% | 99.8% | 99.8% | 980 |
| 1 | 99.6% | 99.8% | 99.7% | 1135 |
| 2 | 99.5% | 99.3% | 99.4% | 1032 |
| 3 | 99.4% | 99.2% | 99.3% | 1010 |
| 4 | 99.3% | 99.5% | 99.4% | 982 |

</div>

Evaluation metrics show consistent performance across all digit classes with minimal confusion between similar digits (e.g., 7 vs 1, 5 vs 6).

---

🛠️ **Tech Stack and Tools Used**

**Frameworks and Libraries**

<div align="center">

| Technology | Version | Purpose |
|-----------|--------|---------|
| TensorFlow | 2.12+ | Deep learning framework |
| Keras | 2.12+ | High-level neural networks API |
| OpenCV | 4.7+ | Computer vision operations |
| PyGame | 2.3+ | GUI and interactive elements |
| NumPy | 1.24+ | Numerical computations |
| Matplotlib | 3.7+ | Visualization and plotting |
| Seaborn | 0.12+ | Statistical data visualization |
| Scikit-learn | 1.2+ | Evaluation metrics |

</div>

**Development Tools**  
- Python 3.8+  
- Jupyter Notebook  
- VS Code / PyCharm  
- Git  
- HTML5/CSS3  

---

🏃‍♂️ **Run Process**  

         ```bash
         git clone https://github.com/yourusername/digit-recognizer.git
         cd digit-recognizer
         pip install -r requirements.txt
         python digit_recognizer.py
Mode selection: Choose between Webcam mode or Drawing Pad mode

Access Settings to configure camera and ROI parameters

Use Evaluate Model to generate performance reports
📋 Requirements
| Package       | Version  | Purpose                       |
| ------------- | -------- | ----------------------------- |
| python        | >=3.8    | Core programming language     |
| tensorflow    | >=2.12.0 | Deep learning framework       |
| opencv-python | >=4.7.0  | Computer vision operations    |
| pygame        | >=2.3.0  | GUI and interactive interface |
| numpy         | >=1.24.0 | Numerical computations        |
| matplotlib    | >=3.7.0  | Data visualization            |
| seaborn       | >=0.12.0 | Statistical visualization     |
| scikit-learn  | >=1.2.0  | Model evaluation metrics      |


💾 Database
MNIST (Modified National Institute of Standards and Technology) dataset:
| Dataset  | Samples | Image Size | Classes  | Format    |
| -------- | ------- | ---------- | -------- | --------- |
| Training | 60,000  | 28×28      | 10 (0-9) | Grayscale |
| Testing  | 10,000  | 28×28      | 10 (0-9) | Grayscale |
| Total    | 70,000  | 28×28      | 10 (0-9) | Grayscale |



<div class="container">

  <!-- User Workflow -->
  <div class="section card">
    <h2>📊 User Workflow</h2>
    <div class="flowchart">
      <pre>
Start
  │
  ↓
Grant Camera Access
  │
  ↓
Load PoseNet Model
  │
  ↓
Model Loaded Successfully
  │
  ↓
Select Exercise Type
  │
  ↓
Begin Pose Estimation
  │
  ↓
Loop [Every Frame @ 30fps]
  │      ├─ Capture Video Frame
  │      ├─ Process Frame in PoseNet
  │      ├─ Return Keypoints Data
  │      ├─ Analyze Exercise Form
  │      ├─ Update Visual Feedback
  │      └─ Display Real-Time Guidance
  │
  ↓
End Session
  │
  ↓
Generate Performance Report
  │
  ↓
Show Results & Progress
  │
  ↓
Exit
      </pre>
    </div>
  </div>

  <!-- System Architecture & Flow -->
  <div class="section card">
    <h2>📊 System Architecture & Flow</h2>
    <div class="flowchart">
      <pre>
Start
  │
  ↓
Load MNIST Dataset
  │
  ↓
Data Preprocessing
  │
  ↓
Build CNN Model
  │
  ↓
Train Model (50 Epochs)
  │
  ↓
Save Model Weights
  │
  ↓
Real-Time Application
  │
  ├─ Webcam Mode ──────┬→ Capture Frame → ROI Selection → Preprocessing → Prediction → Display
  │                     └→ Adjust Parameters → Explainability → Recording
  │
  └─ Drawing Pad Mode ──┬→ Draw Digit → Preprocessing → Prediction → Display
                        └→ Undo/Redo → Save → Explainability
  │
  ↓
Evaluation Mode → Generate Reports → Confusion Matrix → Misclassification Analysis
  │
  ↓
Exit
      </pre>
    </div>
  </div>

  <!-- Future Improvements -->
  <div class="section card">
    <h2>🔮 Future Improvements</h2>
    <div class="grid">
      <div class="card animated-card" style="background: linear-gradient(135deg,#6a11cb,#2575fc); color:#fff;">
        <h3>Recognition</h3>
        <p>Extended character support</p>
        <small>Full alphanumeric capability</small>
      </div>
      <div class="card animated-card" style="background: linear-gradient(135deg,#ff6f00,#ff8e53); color:#fff;">
        <h3>Deployment</h3>
        <p>TensorFlow Lite conversion</p>
        <small>Mobile app deployment</small>
      </div>
      <div class="card animated-card" style="background: linear-gradient(135deg,#12c2e9,#c471ed); color:#fff;">
        <h3>API</h3>
        <p>RESTful endpoints</p>
        <small>Cloud integration</small>
      </div>
      <div class="card animated-card" style="background: linear-gradient(135deg,#f7971e,#ffd200); color:#fff;">
        <h3>Augmentation</h3>
        <p>Style transfer techniques</p>
        <small>Better generalization</small>
      </div>
      <div class="card animated-card" style="background: linear-gradient(135deg,#36d1dc,#5b86e5); color:#fff;">
        <h3>UI/UX</h3>
        <p>Web-based interface</p>
        <small>Broader accessibility</small>
      </div>
      <div class="card animated-card" style="background: linear-gradient(135deg,#ff416c,#ff4b2b); color:#fff;">
        <h3>Collaboration</h3>
        <p>Multi-user support</p>
        <small>Educational applications</small>
      </div>
    </div>
  </div>

  <!-- Creator -->
<div class="section card">
  <h2>👨‍💻 Creator</h2>
  <div class="content">
    <ul style="list-style:none; padding-left:0;">
      <li><strong>Name:</strong> Dibyendu Karmahapatra</li>
      <li><strong>GitHub:</strong> 
        <a href="https://github.com/Dibyendu17122003" target="_blank" 
           style="display:inline-block; padding:0.5rem 1rem; margin-top:0.3rem; background:linear-gradient(135deg,#36d1dc,#5b86e5); color:#fff; text-decoration:none; border-radius:0.5rem;">
           Click here to go to GitHub
        </a>
      </li>
      <li><strong>LinkedIn:</strong> 
        <a href="https://www.linkedin.com/in/dibyendu-karmahapatra-17d2004/" target="_blank" 
           style="display:inline-block; padding:0.5rem 1rem; margin-top:0.3rem; background:linear-gradient(135deg,#36d1dc,#5b86e5); color:#fff; text-decoration:none; border-radius:0.5rem;">
           Click here to go to LinkedIn
        </a>
      </li>
    </ul>
  </div>
</div>



  <!-- Acknowledgments -->
  <div class="section card">
    <h2>🙏 Acknowledgments</h2>
    <div class="content">
      <ul style="list-style:none; padding-left:0;">
        <li>Yann LeCun, Corinna Cortes, and Christopher Burges for the MNIST dataset</li>
        <li>TensorFlow and Keras teams</li>
        <li>PyGame community</li>
        <li>OpenCV contributors</li>
      </ul>
      <p class="center" style="margin-top:1rem;">⭐ Star this repo if you found it helpful!</p>
    </div>
  </div>

</div>


