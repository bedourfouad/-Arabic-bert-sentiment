## 🧠 Arabic Emotion Classification Project

This project focuses on detecting and classifying emotions in Arabic text. It uses the **Emotone\_AR** dataset and applies data preprocessing, exploration, and visualization techniques to prepare the data for machine learning models. The goal is to build an emotion classifier that can identify emotions such as *joy, anger, sadness, fear*, and others from Arabic tweets.

> 🟢 **Live Demo**:
> Try the model on **Hugging Face Spaces** →
> [https://huggingface.co/spaces/bedourfouad/arabic-sentiment-demo](https://huggingface.co/spaces/bedourfouad/arabic-sentiment-demo)

---

### 📁 Dataset

* **Source**: [`emotone-ar`](https://huggingface.co/datasets/emotone-ar-cicling2017/emotone_ar)
* **Language**: Arabic
* **Emotion Classes**:

  * `none` (0)
  * `anger` (1)
  * `joy` (2)
  * `sadness` (3)
  * `love` (4)
  * `sympathy` (5)
  * `surprise` (6)
  * `fear` (7)

---

### 📊 Data Analysis

The notebook performs the following analyses:

* Emotion class distribution (bar and pie charts)
* Text length analysis
* Box plots by emotion category
* Sample tweet display per emotion

---

### 🧹 Text Preprocessing

Cleaning steps include:

* Removing user handles
* Eliminating emojis and special characters
* Formatting Arabic text using:

  * `neattext`
  * `emoji`
  * `pyarabic`
  * `arabic-reshaper`

---

### ⚙️ Installation

Install required libraries using pip:

```bash
pip install datasets neattext emoji pyarabic arabic-reshaper matplotlib seaborn pandas
```

---

### 📦 Project Structure

```bash
.
├── explore_dataset.ipynb     # Jupyter Notebook for analysis and cleaning
├── app/                      # (Optional) Gradio app files for deployment
└── README.md                 # Project documentation
```

---

### 🚀 Deployment

This project is deployed on **Hugging Face Spaces** using the **Gradio** library. Users can enter Arabic text and instantly get the predicted emotion.

🔗 [Live App on Hugging Face](https://huggingface.co/spaces/bedourfouad/arabic-sentiment-demo)

---

### ✅ Future Improvements

* Train deep learning models (CNN, LSTM, BERT)
* Evaluate model using precision, recall, F1-score
* Improve performance with hyperparameter tuning
* Integrate the model into a chatbot or mental health assistant

---

### 👤 Author

Created by **\[Your Name]**
As part of a university graduation project focused on Arabic sentiment and emotion analysis.

---

Let me know if you want this saved as a downloadable `README.md` file or want to include things like:

* A screenshot or GIF of the app
* Hugging Face or Python version badges
* License section (MIT, etc.)
Great! Here's the full **README** file in **English**, ready to use on your GitHub repository:

---

## 🧠 Arabic Emotion Classification Project

This project focuses on detecting and classifying emotions in Arabic text. It uses the **Emotone\_AR** dataset and applies data preprocessing, exploration, and visualization techniques to prepare the data for machine learning models. The goal is to build an emotion classifier that can identify emotions such as *joy, anger, sadness, fear*, and others from Arabic tweets.

> 🟢 **Live Demo**:
> Try the model on **Hugging Face Spaces** →
> [https://huggingface.co/spaces/bedourfouad/arabic-sentiment-demo](https://huggingface.co/spaces/bedourfouad/arabic-sentiment-demo)

---

### 📁 Dataset

* **Source**: [`emotone-ar`](https://huggingface.co/datasets/emotone-ar-cicling2017/emotone_ar)
* **Language**: Arabic
* **Emotion Classes**:

  * `none` (0)
  * `anger` (1)
  * `joy` (2)
  * `sadness` (3)
  * `love` (4)
  * `sympathy` (5)
  * `surprise` (6)
  * `fear` (7)

---

### 📊 Data Analysis

The notebook performs the following analyses:

* Emotion class distribution (bar and pie charts)
* Text length analysis
* Box plots by emotion category
* Sample tweet display per emotion

---

### 🧹 Text Preprocessing

Cleaning steps include:

* Removing user handles
* Eliminating emojis and special characters
* Formatting Arabic text using:

  * `neattext`
  * `emoji`
  * `pyarabic`
  * `arabic-reshaper`

---

### ⚙️ Installation

Install required libraries using pip:

```bash
pip install datasets neattext emoji pyarabic arabic-reshaper matplotlib seaborn pandas
```

---

### 📦 Project Structure

```bash
.
├── explore_dataset.ipynb     # Jupyter Notebook for analysis and cleaning
├── app/                      # (Optional) Gradio app files for deployment
└── README.md                 # Project documentation
```

---

### 🚀 Deployment

This project is deployed on **Hugging Face Spaces** using the **Gradio** library. Users can enter Arabic text and instantly get the predicted emotion.

🔗 [Live App on Hugging Face](https://huggingface.co/spaces/bedourfouad/arabic-sentiment-demo)

---

### ✅ Future Improvements

* Train deep learning models (CNN, LSTM, BERT)
* Evaluate model using precision, recall, F1-score
* Improve performance with hyperparameter tuning
* Integrate the model into a chatbot or mental health assistant

---


