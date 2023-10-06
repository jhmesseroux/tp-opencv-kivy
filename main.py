from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen, ScreenManager

class Screen1(Screen):
    def __init__(self, **kwargs):
        super(Screen1, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        self.label = Label(text='Pantalla 1 - Mensaje 1')
        self.layout.add_widget(self.label)
        self.button = Button(text='Ir a la siguiente pantalla')
        self.button.bind(on_press=self.change_screen)
        self.layout.add_widget(self.button)
        self.add_widget(self.layout)

    def change_screen(self, instance):
        self.manager.current = 'screen2'

class Screen2(Screen):
    def __init__(self, **kwargs):
        super(Screen2, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        self.label = Label(text='Pantalla 2 - Mensaje 2')
        self.layout.add_widget(self.label)
        self.button = Button(text='Ir a la siguiente pantalla')
        self.button.bind(on_press=self.change_screen)
        self.layout.add_widget(self.button)
        self.add_widget(self.layout)

    def change_screen(self, instance):
        self.manager.current = 'screen3'

class Screen3(Screen):
    def __init__(self, **kwargs):
        super(Screen3, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        self.label = Label(text='Pantalla 3 - Mensaje 3')
        self.layout.add_widget(self.label)
        self.add_widget(self.layout)

class TestApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(Screen1(name='screen1'))
        sm.add_widget(Screen2(name='screen2'))
        sm.add_widget(Screen3(name='screen3'))
        return sm

if __name__ == '__main__':
    TestApp().run()
