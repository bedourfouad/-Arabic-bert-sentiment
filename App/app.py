# app.py - تطبيق Gradio لتحليل المشاعر العربية (بدون عرض النسب)

import gradio as gr
from transformers import pipeline
import torch
import os
import time

# --- 1. تحديد اسم مستودع النموذج على Hugging Face Hub ---
repo_id_model = "bedourfouad/arabic-bert-sentiment" # تأكدي أنه اسم المستودع الصحيح

# --- 2. تحميل الـ pipeline ---
print(f"Loading model pipeline: {repo_id_model}...")
start_time = time.time()
classifier = None
load_error = None

try:
    device_num = 0 if torch.cuda.is_available() else -1
    if device_num == 0:
        print("CUDA (GPU) is available, using device 0.")
    else:
        print("CUDA (GPU) not available, using CPU (device -1).")

    # top_k=1 للحصول على أفضل تصنيف فقط
    classifier = pipeline(
        "text-classification",
        model=repo_id_model,
        tokenizer=repo_id_model,
        device=device_num,
        top_k=1 # <<<--- تغيير مهم: احصل على أفضل تصنيف واحد فقط
    )
    load_time = time.time() - start_time
    print(f"✅ Pipeline loaded successfully in {load_time:.2f} seconds.")
except Exception as e:
    load_error = e
    print(f"❌ Error loading pipeline: {e}")
    print("   Interface will show an error message.")

# --- 3. تعريف دالة التنبؤ للواجهة ---
def predict_sentiment_gradio_label_only(text_input):
    """
    تأخذ نص الإدخال من Gradio وتُرجع اسم المشاعر الأعلى احتمالاً كنص.
    """
    if classifier is None:
        return f"خطأ في تحميل النموذج: {load_error}"

    if not text_input or not isinstance(text_input, str) or text_input.strip() == "":
        return "الرجاء إدخال نص."

    print(f"\nPredicting for text: '{text_input[:50]}...'")
    start_predict_time = time.time()
    try:
        # الـ pipeline مع top_k=1 يرجع قائمة تحتوي على قاموس واحد
        results = classifier(text_input)
        predict_time = time.time() - start_predict_time
        print(f"Prediction completed in {predict_time:.2f} seconds.")

        if results and results[0]:
            # results[0] هو قائمة، ونتيجتنا في results[0][0]
            predicted_label = results[0][0]['label'] # اسم المشاعر الأعلى
            print(f"Predicted label: {predicted_label}")
            return predicted_label # إرجاع اسم المشاعر فقط
        else:
            print("Prediction returned empty results.")
            return "لم يتمكن النموذج من التنبؤ."
    except Exception as e:
        print(f"❌ Error during prediction function: {e}")
        return f"خطأ أثناء التنبؤ: {e}"

# --- 4. إنشاء واجهة Gradio ---
description = f"""
تحليل المشاعر للنصوص العربية باستخدام نموذج BERT مُعدل ({repo_id_model}).
أدخل نصًا في المربع أدناه واضغط على "Submit" لرؤية المشاعر المتوقعة.
المشاعر الممكنة هي:
anger (غضب), fear (خوف), joy (فرح), love (حب), none (محايد/لا شيء), sadness (حزن), surprise (مفاجأة), sympathy (تعاطف).
"""
examples = [
    ["أنا سعيد جدًا بهذا اليوم الرائع!"],
    ["كم هو محبط هذا الموقف، أشعر بالغضب."],
    ["لم أتوقع هذا الخبر أبداً، يا للمفاجأة!"],
]

with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown(f"# 🇸🇦 تحليل المشاعر للنصوص العربية")
    gr.Markdown(description)
    with gr.Row():
        input_textbox = gr.Textbox(
            lines=5,
            label="النص العربي",
            placeholder="اكتب النص هنا لتحليل المشاعر..."
        )
        # تغيير المخرج إلى gr.Text لعرض نص بسيط
        output_text = gr.Text(label="المشاعر المتوقعة") # <<<--- تغيير مهم
    submit_button = gr.Button("تحليل المشاعر (Submit)")
    gr.Examples(examples=examples, inputs=input_textbox)

    submit_button.click(
        fn=predict_sentiment_gradio_label_only, # استخدام الدالة الجديدة
        inputs=input_textbox,
        outputs=output_text, # المخرج هو مكون النص الجديد
        api_name="generate_sentiment_label" # يمكن تغيير اسم الـ API endpoint
    )

# --- 5. تشغيل الواجهة ---
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0")