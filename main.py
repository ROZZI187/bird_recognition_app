from kivy.app import App
from kivy.clock import Clock
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.image import Image
from kivymd.app import MDApp
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
from kivymd.uix.screen import MDScreen
from kivy.core.window import Window
from tensorflow.lite.python.interpreter import Interpreter
from datetime import datetime
import os
import librosa
import soundfile as sf
from jnius import autoclass
from android.permissions import request_permissions, Permission, check_permission
from tensorflow.image import resize
import numpy as np

class SplashScreen(Screen):
    def __init__(self, **kwargs):
        super(SplashScreen, self).__init__(**kwargs)
        self.image = Image(source="splash_screen.png", allow_stretch=True, keep_ratio=False)
        self.image.size = Window.size
        self.add_widget(self.image)
        Clock.schedule_once(self.change_screen, 10)

    def change_screen(self, dt):
        app = MDApp.get_running_app()
        app.screen_manager.current = "main"

class BirdRecApp(MDApp):
    def __init__(self, **kwargs):
        super(BirdRecApp, self).__init__(**kwargs)
        self.is_recording = False
        self.filename = None
        self.interpreter = None
        self.dark_mode = False

    def build(self):
        self.screen_manager = ScreenManager()
        splash_screen = SplashScreen()
        self.screen_manager.add_widget(splash_screen)
        self.main_screen = MDScreen(name='main')
        self.screen_manager.add_widget(self.main_screen)
        if not os.path.isdir("/sdcard/kivyrecords/"):
            os.mkdir("/sdcard/kivyrecords/")
        model_path = 'model_666.tflite'  
        self.interpreter = Interpreter(model_path)
        self.interpreter.allocate_tensors()
        return self.screen_manager
    
    def get_audio_file_path(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"/sdcard/kivyrecords/recording_{timestamp}.wav"
    
    def start_audio_recording(self):
        MediaRecorder = autoclass('android.media.MediaRecorder')
        AudioSource = autoclass('android.media.MediaRecorder$AudioSource')
        OutputFormat = autoclass('android.media.MediaRecorder$OutputFormat')
        AudioEncoder = autoclass('android.media.MediaRecorder$AudioEncoder')

        self.recorder = MediaRecorder()
        self.recorder.setAudioSource(AudioSource.MIC)
        self.recorder.setOutputFormat(OutputFormat.THREE_GPP)
        self.recorder.setAudioEncoder(AudioEncoder.AMR_NB)
        self.recorder.setOutputFile(self.filename)
        self.recorder.prepare()
        self.recorder.start()

    def stop_audio_recording(self):
        if hasattr(self, 'recorder'):
            self.recorder.stop()
            self.recorder.release()

    def record_audio(self):
        if self.is_recording:
            self.start_audio_recording()
        else:
            self.stop_audio_recording()
            self.show_recognition_alert()

    def show_recognition_alert(self, *args):
        if self.filename and os.path.exists(self.filename):
            audio_data, sample_rate = sf.read(self.filename)

            mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
            mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)
            mel_spectrogram_resized = resize(mel_spectrogram_db[np.newaxis, ..., np.newaxis], [200, 200])

            input_data = mel_spectrogram_resized.numpy().astype(np.float32)

            self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], input_data)
            self.interpreter.invoke()

            predictions = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])[0]
            bird_class_index = np.argmax(predictions)
            bird_classes = [
                'bazant', 'bielik', 'bocian_czarny', 'czapla_siwa', 'czeczotka_brazowa', 'dudek',
                'dzieciol_bialogrzebiety', 'gil', 'jastrzab', 'jerzyk', 'kania_ruda', 'kos', 'kruk',
                'krzyzowka', 'kulik_mniejszy', 'kwiczol', 'mysikrolik', 'myszolow', 'orlik_krzykliwy',
                'pliszka_siwa', 'pustulka_zwyczajna', 'rudzik', 'slowik_rdzawy', 'sojka', 'sroka',
                'szpak', 'wrona_siwa', 'zieba', 'zuraw'
            ]
            recognized_bird = bird_classes[bird_class_index]
            probability = np.max(predictions) * 100  

            dialog = MDDialog(
                text=f"Rozpoznano ptaka: {recognized_bird}\nPrawdopodobieństwo rozpoznania: {probability}%",
                size_hint=(0.8, 0.2),
                buttons=[
                    MDFlatButton(
                        text="OK",
                        on_release=lambda x: self.close_recognition_dialog(dialog)
                    )
                ]
            )
            dialog.open()

    def close_recognition_dialog(self, dialog):
        dialog.dismiss()
        self.screen_manager.current = 'main'

    def update_timer(self, dt):
        self.record_time += 1
        minutes = self.record_time // 60
        seconds = self.record_time % 60
        self.main_screen.ids.timer_label.text = f"{minutes:02}:{seconds:02}"

    def toggle_dark_mode(self):
        if self.theme_cls.theme_style == "Light":
            self.theme_cls.theme_style = "Dark"
            self.dark_mode = True
        else:
            self.theme_cls.theme_style = "Light"
            self.dark_mode = False

if __name__ == '__main__':
    BirdRecApp().run()
