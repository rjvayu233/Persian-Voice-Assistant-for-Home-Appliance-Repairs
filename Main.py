import time
import wave
import json
import streamlit as st
import pyaudio
import os
from vosk import Model, KaldiRecognizer
import tempfile
import re

# --- کتابخانه‌های صدا ---
import sounddevice as sd
import soundfile as sf
import torch
from transformers import pipeline
from scipy.io.wavfile import write as write_wav

# --- کتابخانه‌های نقشه ---
import folium
from streamlit_folium import st_folium
import pandas as pd

# --- (اصلاح شد) مسیر فایل نقشه آفلاین ---
# لطفا فایل قبلی را پاک کرده و این فایل جدید را دانلود کنید:
# https://raw.githubusercontent.com/datasets/geo-boundaries-irn/master/iran-provinces.geojson
GEOJSON_PATH = r"C:\Users\FaraCom\Desktop\chatbot\iran-provinces.geojson"


# --- بخش ۱: بارگذاری مدل‌ها (بدون تغییر) ---

def load_tts_model():
    if 'tts_pipeline' not in st.session_state:
        print("در حال آماده‌سازی مدل صوتی (MMS)... ⏳")
        try:
            device = 0 if torch.cuda.is_available() else -1
            print(f"Using device: {'cuda' if device == 0 else 'cpu'}")
            st.session_state.tts_pipeline = pipeline(
                "text-to-speech", model="facebook/mms-tts-fas", device=device
            )
            print("Transformers MMS model loaded successfully.")
        except Exception as e:
            print(f"Error loading Transformers MMS model: {e}")
            st.session_state.tts_pipeline = None


def load_vosk_model():
    if 'vosk_model' not in st.session_state:
        print("در حال آماده‌سازی مدل تشخیص گفتار (Vosk)... ⏳")
        large_model_path = r"C:\Users\FaraCom\Desktop\chatbot\vosk-model-fa-0.5"
        small_model_path = r"C:\Users\FaraCom\Desktop\chatbot\vosk-model-small-fa-0.42"
        model = None
        try:
            if os.path.exists(os.path.join(large_model_path, "am")):
                model = Model(large_model_path)
                print("Using Large Vosk model.")
            else:
                print("Model بزرگ Vosk یافت نشد. (لطفاً برای دقت بالا نصب کنید)")
                raise FileNotFoundError("Large model not found")
        except Exception:
            print("مدل بزرگ Vosk لود نشد. استفاده از مدل کوچک.")
            try:
                if os.path.exists(os.path.join(small_model_path, "am")):
                    model = Model(small_model_path)
                    print("Using Small Vosk model.")
                else:
                    raise FileNotFoundError("Small model not found")
            except Exception as e_small:
                print(f"خطای حیاتی: هیچکدام از مدل‌های Vosk یافت نشدند: {e_small}")
                model = None
        st.session_state.vosk_model = model
        if model:
            print("مدل تشخیص گفتار (Vosk) با موفقیت بارگذاری شد.")


# --- بخش ۲: ابزارهای صدا (بدون تغییر) ---

def speak(text):
    if 'tts_pipeline' not in st.session_state or st.session_state.tts_pipeline is None:
        st.error("مدل TTS (MMS) بارگذاری نشده است.")
        return
    print(f"Assistant speaking (Transformers MMS): {text}")
    try:
        pipeline = st.session_state.tts_pipeline
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            filename = f.name
        with st.spinner("... 🤖 در حال تولید صدا (آفلاین) ..."):
            output = pipeline(text)
            write_wav(filename, rate=output["sampling_rate"], data=output["audio"].squeeze())
        data, fs = sf.read(filename, dtype='float32')
        sd.play(data, fs)
        duration_in_seconds = len(data) / fs
        time.sleep(duration_in_seconds + 0.3)
        os.remove(filename)
    except Exception as e:
        print(f"Error in Transformers MMS speak: {e}")


def record_audio(filename, duration=5):
    # ... (کد بدون تغییر) ...
    recognizer = pyaudio.PyAudio()
    stream = recognizer.open(format=pyaudio.paInt16, channels=1, rate=16000,
                             input=True, frames_per_buffer=1024)
    print(f"Recording for {duration} seconds...")
    frames = []
    for i in range(0, int(16000 / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    recognizer.terminate()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(recognizer.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()


def recognize_audio_vosk(filename):
    # ... (کد بدون تغییر) ...
    if 'vosk_model' not in st.session_state or st.session_state.vosk_model is None:
        st.error("مدل Vosk بارگذاری نشده است.")
        return ""
    model = st.session_state.vosk_model
    wf = wave.open(filename, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)
    result = rec.FinalResult()
    return json.loads(result)["text"]


# --- بخش ۳: هوشمندسازی و دیتابیس (بازگشت به کلید فارسی) ---

def extract_name(text):
    # ... (کد بدون تغییر) ...
    intro_phrases = ["با سلام", "سلام", "وقت بخیر", "درود", "خسته نباشید", "هستم", "هسدم", "میباشم", "اسم من", "از"]
    pattern = r'(' + '|'.join(re.escape(p) for p in intro_phrases) + r')'
    clean_text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip(" .,،")
    return clean_text if clean_text else text


# --- (دیتابیس کامل ۳۱ استان با کلید فارسی) ---
PROVINCE_KEYWORDS = {
    "آذربایجان شرقی": ["آذربایجان شرقی", "تبریز"], "آذربایجان غربی": ["آذربایجان غربی", "ارومیه"],
    "اردبیل": ["اردبیل"], "اصفهان": ["اصفهان"], "البرز": ["البرز", "کرج"], "ایلام": ["ایلام"],
    "بوشهر": ["بوشهر"], "تهران": ["تهران"], "چهارمحال و بختیاری": ["چهارمحال و بختیاری", "شهرکرد"],
    "خراسان جنوبی": ["خراسان جنوبی", "بیرجند"], "خراسان رضوی": ["خراسان رضوی", "مشهد"],
    "خراسان شمالی": ["خراسان شمالی", "بجنورد"], "خوزستان": ["خوزستان", "اهواز"], "زنجان": ["زنجان"],
    "سمنان": ["سمنان"], "سیستان و بلوچستان": ["سیستان و بلوچستان", "زاهدان"], "فارس": ["فارس", "شیراز"],
    "قزوین": ["قزوین"], "قم": ["قم"], "کردستان": ["کردستان", "سنندج"], "کرمان": ["کرمان"],
    "کرمانشاه": ["کرمانشاه"], "کهگیلویه و بویراحمد": ["کهگیلویه و بویراحمد", "یاسوج"],
    "گلستان": ["گلستان", "گرگان"], "گیلان": ["گیلان", "رشت"], "لرستان": ["لرستان", "خرم آباد"],
    "مازندران": ["مازندران", "ساری"], "مرکزی": ["مرکزی", "اراک"], "هرمزگان": ["هرمزگان", "بندرعباس"],
    "همدان": ["همدان"], "یزد": ["یزد"]
}
TEHRAN_AREAS = ["ونک", "جردن", "سعادت آباد", "شهرک غرب", "تهرانپارس", "پیروزی", "نارمک"]
KARAJ_AREAS = ["گوهردشت", "عظیمیه", "مهرشهر", "جهانشهر", "فردیس", "باغستان", "گلشهر"]

PROVINCE_COORDS = {
    "آذربایجان شرقی": [38.08, 46.29], "آذربایجان غربی": [37.55, 45.07], "اردبیل": [38.25, 48.29],
    "اصفهان": [32.65, 51.68], "البرز": [35.83, 50.93], "ایلام": [33.64, 46.42],
    "بوشهر": [28.97, 50.84], "تهران": [35.69, 51.39], "چهارمحال و بختیاری": [32.33, 50.85],
    "خراسان جنوبی": [32.86, 59.22], "خراسان رضوی": [36.30, 59.60], "خراسان شمالی": [37.47, 57.33],
    "خوزستان": [31.32, 48.69], "زنجان": [36.68, 48.48], "سمنان": [35.57, 53.38],
    "سیستان و بلوچستان": [29.49, 60.86], "فارس": [29.61, 52.53], "قزوین": [36.27, 50.00],
    "قم": [34.64, 50.88], "کردستان": [35.31, 46.99], "کرمان": [30.28, 57.08],
    "کرمانشاه": [34.31, 47.06], "کهگیلویه و بویراحمد": [30.67, 51.59], "گلستان": [36.84, 54.44],
    "گیلان": [37.28, 49.59], "لرستان": [33.49, 48.35], "مازندران": [36.57, 53.06],
    "مرکزی": [34.09, 49.69], "هرمزگان": [27.19, 56.28], "همدان": [34.79, 48.51],
    "یزد": [31.89, 54.37]
}


def extract_location(text):
    # ... (کد بدون تغییر) ...
    found_province = None
    found_area = None
    for prov, keywords in PROVINCE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                found_province = prov
                break
        if found_province: break
    for area in KARAJ_AREAS:
        if area in text:
            found_province = "البرز"
            found_area = area
            break
    if not found_area:
        for area in TEHRAN_AREAS:
            if area in text:
                found_province = "تهران"
                found_area = area
                break
    if not found_province and not found_area:
        return "نا مشخص", "نا مشخص", None

    display_province = found_province
    display_area = found_area if found_area else "مرکز استان"

    return display_province, display_area, found_province  # <-- کلید فارسی را برمی‌گردانیم


QA_DATABASE = {
    # ... (دیتابیس بدون تغییر) ...
    "یخچال": {"keywords": ["یخچال", "فریزر", "ساید", "سرد", "یخساز", "برد", "سوخته", "خنک نمیکنه", "صدا میده", "برفک"],
              "سوالات": {"خنک نمیکنه": "خنک نکردن", "صدا میده": "صدای غیرعادی", "برفک": "برفک زدن",
                         "روشن نمیشه": "روشن نشدن"}},
    "ماشین لباسشویی": {"keywords": ["لباسشویی", "شستشو", "خشک کن", "نمیچرخه", "آب نمیده", "تخلیه", "صدا میده"],
                       "سوالات": {"روشن نمیشه": "روشن نشدن", "آب تخلیه نمیکنه": "عدم تخلیه آب",
                                  "نمیچرخه": "نچرخیدن دیگ"}},
    "تلویزیون": {"keywords": ["تلویزیون", "تی وی", "صفحه", "نمایش", "تصویر"],
                 "سوالات": {"روشن نمیشه": "روشن نشدن", "صدا نداره": "قطعی صدا", "تصویر نداره": "قطعی تصویر"}},
    "کولر": {"keywords": ["کولر", "گازی", "تهویه", "باد", "خنک", "اسپلیت"],
             "سوالات": {"خنک نمیکنه": "خنک نکردن", "باد نمیزنه": "نداشتن پرتاب باد", "روشن نمیشه": "روشن نشدن"}},
    "اجاق گاز": {"keywords": ["گاز", "اجاق", "فر"],
                 "سوالات": {"روشن نمیشه": "روشن نشدن شعله", "جرقه": "جرقه نزدن", "فر": "خرابی فر"}},
    "مایکروویو": {"keywords": ["مایکروویو", "مایکروفر"],
                  "سوالات": {"گرم نمیکنه": "گرم نکردن", "روشن نمیشه": "روشن نشدن", "نمیچرخه": "نچرخیدن سینی"}},
    "جاروبرقی": {"keywords": ["جاروبرقی", "جارو", "مکش"],
                 "سوالات": {"مکش": "ضعیف بودن مکش", "روشن نمیشه": "روشن نشدن", "صدا": "صدای زیاد"}}
}


def extract_details(text):
    # ... (کد بدون تغییر) ...
    detected_device = None
    detected_problem = None
    for device, data in QA_DATABASE.items():
        for keyword in data["keywords"]:
            if keyword in text:
                detected_device = device
                break
        if detected_device:
            break
    if not detected_device:
        return "دستگاه نامشخص", text if text else "مشکل نامشخص"
    device_data = QA_DATABASE[detected_device]
    for problem_keyword, problem_desc in device_data["سوالات"].items():
        if problem_keyword in text:
            detected_problem = problem_desc
            break
    if not detected_problem:
        detected_problem = "مشکل عمومی"
    return detected_device, detected_problem


# --- بخش ۴: رابط کاربری و فرم (بدون تغییر) ---

def apply_custom_css():
    st.markdown("""
        <style>
            body { 
                font-family: 'Tahoma', sans-serif; 
                direction: rtl; 
            }
            .title {
                text-align: center;
                font-size: 32px;
                font-weight: bold;
                margin-bottom: 30px;
            }
            .stButton>button {
                width: 100%;
                font-size: 16px;
                padding: 10px;
                font-weight: bold;
            }
            .chat-bubble {
                padding: 12px 18px;
                border-radius: 18px;
                margin-bottom: 10px;
                max-width: 75%;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                font-size: 16px;
                line-height: 1.6;
                width: fit-content;
            }
            .user-bubble {
                background-color: #e1ffc7; 
                text-align: right;
                margin-left: auto;
                color: #000000; 
            }
            .assistant-bubble {
                background-color: #f0f0f0; 
                border: 1px solid #f0f0f0;
                text-align: right;
                margin-right: auto;
                color: #000000; 
            }
            .form-container {
                width: 100%;
                margin: 0 auto;
                background-color: #ffffff;
                border: 2px solid #2ecc71;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                direction: rtl;
            }
            .form-title { font-size: 20px; font-weight: bold; color: #2c3e50; text-align: center; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 15px; }
            .form-field { font-size: 16px; margin-bottom: 10px; color: #2c3e50; }
            .form-field span { font-weight: bold; color: #2980b9; margin-left: 10px; }

            .center-map {
                display: flex;
                justify-content: center;
                width: 100%;
            }
        </style>
    """, unsafe_allow_html=True)


def display_submission_form(name, location, request):
    # ... (کد بدون تغییر) ...
    name_display = name if name else "..."
    location_display = location if location else "..."
    request_display = request if request else "..."
    st.markdown(f"""
    <div class="form-container">
        <div class="form-title">📋 فرم ثبت درخواست</div>
        <div class="form-field"><span>👤 نام و فامیل:</span> {name_display}</div>
        <div class="form-field"><span>📍 موقعیت:</span> {location_display}</div>
        <div class="form-field"><span>📝 درخواست:</span> {request_display}</div>
    </div>
    """, unsafe_allow_html=True)


# --- (تابع نمایش نقشه اصلاح‌شده) ---
def display_map(highlight_province_key):
    """
    (حل مشکل ۱ و ۲)
    نقشه ایران را با زوم (7) روی استان هایلایت شده (بر اساس کلید فارسی) نمایش می‌دهد.
    """

    @st.cache_data
    def get_geojson():
        if not os.path.exists(GEOJSON_PATH):
            print(f"خطا: فایل نقشه در مسیر '{GEOJSON_PATH}' یافت نشد.")
            return None
        try:
            with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"خطا در خواندن فایل GeoJSON: {e}")
            return None

    geojson_data = get_geojson()
    if not geojson_data:
        return  # اگر نقشه لود نشد، ادامه نده

    # --- (حل مشکل ۱: زوم) ---
    if highlight_province_key in PROVINCE_COORDS:
        map_location = PROVINCE_COORDS[highlight_province_key]
        map_zoom = 7  # <-- زوم کمتر شد
    else:
        map_location = [32.4279, 53.6880]
        map_zoom = 4.5
    # --- پایان اصلاح ---

    try:
        m = folium.Map(location=map_location, zoom_start=map_zoom)

        all_provinces = list(PROVINCE_KEYWORDS.keys())  # ["آذربایجان شرقی", ...]
        df_data = pd.DataFrame(all_provinces, columns=["name"])
        df_data['value'] = 0

        # --- (حل مشکل ۲: هایلایت) ---
        if highlight_province_key in df_data['name'].values:
            df_data.loc[df_data['name'] == highlight_province_key, 'value'] = 1

        folium.Choropleth(
            geo_data=geojson_data,
            name="choropleth",
            data=df_data,
            columns=["name", "value"],
            key_on="feature.properties.name",  # <-- بازگشت به 'name'
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.5,
            legend_name="موقعیت انتخاب شده",
            highlight=True,
            nan_fill_color="white",
            nan_fill_opacity=0.1,
        ).add_to(m)

        folium.GeoJson(
            geojson_data,
            tooltip=folium.GeoJsonTooltip(
                fields=["name"],  # <-- بازگشت به 'name'
                aliases=["استان:"],
                style=("background-color: white; color: black; font-family: Tahoma;"),
            )
        ).add_to(m)

        st_folium(m, width=400, height=300, returned_objects=[])

    except Exception as e:
        # st.error(f"خطا در ساخت نقشه: {e}") # <-- (حل مشکل ۱: حذف ارور)
        print(f"Error rendering map: {e}")


# --- بخش ۵: تابع اصلی برنامه (UI اصلاح‌شده) ---

def main():
    st.set_page_config(page_title="دستیار صوتی تعمیرات", page_icon="🛠️", layout="wide")
    apply_custom_css()

    st.markdown("<h1 style='text-align: center; direction: rtl;'>🛠️ دستیار صوتی تعمیرات لوازم خانگی 🎙️</h1>",
                unsafe_allow_html=True)
    st.markdown("---")

    # لودرها به صورت پنهان اجرا می‌شوند
    load_tts_model()
    load_vosk_model()

    if 'stage' not in st.session_state:
        st.session_state.stage = "get_name"
        st.session_state.name = None
        st.session_state.location_raw = None
        st.session_state.location_processed = None
        st.session_state.location_key = None
        st.session_state.issue_text_raw = None
        st.session_state.main_issue_processed = None
        st.session_state.chat_history = []
        st.session_state.prompted_done = False

    if not st.session_state.chat_history:
        # --- حالت تک ستونه (فقط در شروع) ---
        _, col_center, _ = st.columns([1, 2, 1])
        with col_center:
            st.subheader("کنترل پنل")
            st.info("برای شروع، دکمه زیر را بزنید...")
            if st.button("🎙️ شروع و ضبط نام (۵ ثانیه)"):
                prompt_text = "سلام وقتتون بخیر. برای ثبت درخواست، لطفاً نام و نام خانوادگی خود را بگویید."
                st.session_state.chat_history.append(("assistant", prompt_text))
                speak(prompt_text)
                with st.spinner("... در حال شنیدن ..."):
                    record_audio("name_audio.wav", duration=5)
                with st.spinner("... در حال پردازش نام ..."):
                    name_text = recognize_audio_vosk("name_audio.wav")
                if name_text:
                    st.session_state.name_raw = name_text
                    st.session_state.name = extract_name(name_text)
                    st.session_state.chat_history.append(("user", name_text))
                    st.session_state.stage = "get_location"
                    st.rerun()
                else:
                    st.error("صدایی تشخیص داده نشد. لطفا دوباره دکمه را بزنید.")

    else:
        # --- حالت دو ستونه (پس از شروع) ---
        col_main_ui, col_chat = st.columns([1, 2])  # کنترل پنل (راست)، چت (چپ)

        with col_chat:
            st.subheader("تاریخچه گفتگو")
            with st.container(height=500):
                for speaker, text in st.session_state.chat_history:
                    if speaker == "user":
                        st.markdown(f'<div class="chat-bubble user-bubble">{text}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-bubble assistant-bubble">{text}</div>', unsafe_allow_html=True)

        with col_main_ui:
            st.subheader("کنترل پنل")

            if st.session_state.stage == "get_location":
                st.info(f"متشکرم {st.session_state.name}. دکمه زیر را بزنید و آدرس خود را بگویید.")
                if st.button("🎙️ ضبط استان و منطقه (۵ ثانیه)"):
                    prompt_text = "لطفاً استان و نام منطقه‌ای که در آن زندگی می‌کنید را بگویید. (مثلاً: کرج، گوهردشت یا تهران، ونک)"
                    st.session_state.chat_history.append(("assistant", prompt_text))
                    speak(prompt_text)
                    with st.spinner("... در حال شنیدن موقعیت ..."):
                        record_audio("location_audio.wav", duration=5)
                    with st.spinner("... در حال پردازش موقعیت ..."):
                        location_text = recognize_audio_vosk("location_audio.wav")
                    if location_text:
                        st.session_state.location_raw = location_text
                        prov_fa, area_fa, prov_key = extract_location(location_text)
                        st.session_state.location_processed = f"{prov_fa}، {area_fa}"
                        st.session_state.location_key = prov_key
                        st.session_state.chat_history.append(("user", location_text))
                        st.session_state.stage = "get_issue"
                        st.rerun()
                    else:
                        st.error("صدایی تشخیص داده نشد. لطفا دوباره تلاش کنید.")

                if st.button("🔄 ضبط مجدد نام"):
                    st.session_state.stage = "get_name"
                    st.session_state.name = None
                    st.session_state.name_raw = None
                    st.session_state.chat_history.clear()
                    st.rerun()


            elif st.session_state.stage == "get_issue":
                st.info(f"عالی. دکمه زیر را بزنید و مشکل دستگاه را بگویید.")
                if st.button("🎙️ ضبط مشکل (۷ ثانیه)"):
                    prompt_text = f"سلام {st.session_state.name}، مشکل دستگاهتون چیه؟"
                    st.session_state.chat_history.append(("assistant", prompt_text))
                    speak(prompt_text)
                    with st.spinner("... در حال شنیدن مشکل ..."):
                        record_audio("issue_audio.wav", duration=7)
                    with st.spinner("... در حال پردازش ..."):
                        issue_text = recognize_audio_vosk("issue_audio.wav")
                    if issue_text:
                        st.session_state.issue_text_raw = issue_text
                        device, problem = extract_details(issue_text)
                        st.session_state.main_issue_processed = f"{device} (مشکل: {problem})"
                        st.session_state.chat_history.append(("user", issue_text))
                        st.session_state.stage = "done"
                        st.rerun()
                    else:
                        st.error("صدایی تشخیص داده نشد. لطفا دوباره تلاش کنید.")

                if st.button("🔄 ضبط مجدد موقعیت"):
                    st.session_state.stage = "get_location"
                    st.session_state.location_raw = None
                    st.session_state.location_processed = None
                    st.session_state.location_key = None
                    if len(st.session_state.chat_history) >= 2:
                        st.session_state.chat_history.pop()
                        st.session_state.chat_history.pop()
                    st.rerun()

            elif st.session_state.stage == "done":
                if not st.session_state.prompted_done:
                    final_message = f"متشکرم. درخواست شما برای «{st.session_state.main_issue_processed}» ثبت شد. تیم پشتیبانی به زودی با شما (از {st.session_state.location_processed}) تماس خواهند گرفت."
                    st.session_state.chat_history.append(("assistant", final_message))
                    speak(final_message)
                    st.session_state.prompted_done = True
                    st.rerun()

                if st.session_state.prompted_done:
                    st.success(f"✅ درخواست شما با موفقیت ثبت شد.")
                    st.markdown("---")
                    if st.button("🔄 ثبت یک درخواست جدید"):
                        st.session_state.clear()
                        st.rerun()

                    if st.button("🔄 ضبط مجدد مشکل"):
                        st.session_state.stage = "get_issue"
                        st.session_state.issue_text_raw = None
                        st.session_state.main_issue_processed = None
                        st.session_state.prompted_done = False
                        if len(st.session_state.chat_history) >= 2:
                            st.session_state.chat_history.pop()
                            st.session_state.chat_history.pop()
                        st.rerun()

                    st.markdown("**خلاصه درخواست:**")
                    display_submission_form(st.session_state.name,
                                            st.session_state.location_processed,
                                            st.session_state.main_issue_processed)

                    st.markdown("---")
                    st.markdown(f"**موقعیت {st.session_state.name}:**")
                    display_map(st.session_state.location_key)

                    # --- (حل مشکل نقشه) ---
    # بلوک پایانی نقشه حذف شد.


if __name__ == "__main__":

    main()
