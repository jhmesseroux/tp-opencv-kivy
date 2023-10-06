from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen, ScreenManager
import cv2, numpy, sys, os, time

class Screen1(Screen):
    def __init__(self, **kwargs):
        super(Screen1, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        self.label = Label(text='Inicio de sesion y creacion de cuenta mediante reconocimiento facial')
        self.layout.add_widget(self.label)
        self.button = Button(text='Ir a la siguiente pantalla')
        self.btnSignUp = Button(text='Registrarse')
        self.btnSignIp = Button(text='Inicio de sesion')
        self.button.bind(on_press=self.change_screen)
        self.btnSignUp.bind(on_press=self.register)
        self.btnSignIp.bind(on_press=self.SignIn)
        self.layout.add_widget(self.button)
        self.layout.add_widget(self.btnSignIp)
        self.layout.add_widget(self.btnSignUp)
        self.add_widget(self.layout)

    def change_screen(self, instance):
        self.manager.current = 'screen2'
    def register(self,instance):
        # change the paths below to the location where these files are on your machine
        # haar_file = '/path/to/project/directory/haarcascade_frontalface_default.xml'
        haar_file = 'C:/Users/jhmes/Documents/projects/python/tpopencv/haarcascade_frontalface_default.xml'

        # All of the faces data (images) will be stored here
        datasets = 'C:/Users/jhmes/Documents/projects/python/tpopencv/faces'
        # Sub dataset in 'faces' folder. Each folder is specific to an individual person
        # change the name below when creating a new dataset for a new person
        sub_dataset = 'test_user'

        # join the paths to include the sub_dataset folder
        path = os.path.join(datasets, sub_dataset)
        # if sub_dataset folder doesn't already exist, make the folder with the name defined above
        if not os.path.isdir(path):
            os.mkdir(path)

        # defining the size of images
        (width, height) = (130, 100)

        face_cascade = cv2.CascadeClassifier(haar_file)
        # use '0' for internal (built-in) webcam or '1' for external ones
        webcam = cv2.VideoCapture(0)
        # returns true or false (if the camera is on or not)
        print("Webcam is open? ", webcam.isOpened())
        # wait for the camera to turn on (just to be safe, in case the camera needs time to load up)
        time.sleep(2)
        #Takes pictures of detected face and saves them
        count = 1
        print("Taking pictures...")
        # this takes 100 pictures of your face. Change this number if you want.
        # Having too many images, however, might slow down the program
        while count < 202:
            # im = camera stream
            ret_val, im = webcam.read()
            # if it recieves something from the webcam...
            if ret_val == True:
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                # detect face using the haar cascade file
                faces = face_cascade.detectMultiScale(gray, 1.3, 4)
                for (x,y,w,h) in faces:
                    # draws a rectangle around your face when taking pictures
                    # this is to create a ROI (region of interest) so it only takes pictures of your face
                    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                    # define 'face' as the inside of the rectangle we made above and make it grayscale
                    face = gray[y:y + h, x:x + w]
                    # resize the face images to the size of the 'face' variable above (i.e: area captured inside of the rectangle)
                    face_resize = cv2.resize(face, (width, height))
                    # save images with their corresponding number
                    cv2.imwrite('%s/%s.png' % (path,count), face_resize)
                count += 1
                # display the openCV window
                cv2.imshow('OpenCV', im)
                key = cv2.waitKey(20)
                # press esc to stop the loop
                if key == 27:
                    break
        print("Sub dataset for your face has been created.")
        webcam.release()
        cv2.destroyAllWindows()

    def SignIn(Self,instance):
        haar_file = 'C:/Users/jhmes/Documents/projects/python/tpopencv/haarcascade_frontalface_default.xml'# path to the main faces directory which contains all the sub_datasets
        datasets = 'C:/Users/jhmes/Documents/projects/python/tpopencv/faces'

        print('Training classifier...')
        # Create a list of images and a list of corresponding names along with a unique id
        (images, labels, names, id) = ([], [], {}, 0)
        for (subdirs, dirs, files) in os.walk(datasets):
            for subdir in dirs:
            # the person's name is the name of the sub_dataset created using the create_data.py file
                names[id] = subdir
                subjectpath = os.path.join(datasets, subdir)
                for filename in os.listdir(subjectpath):
                    path = subjectpath + '/' + filename
                    label = id
                    images.append(cv2.imread(path, 0))
                    labels.append(int(label))
                id += 1
        (width, height) = (130, 100)

        # https://www.life2coding.com/drawing-fancy-round-rectangle-using-opencv-python/
        def rounded_rectangle(img, pt1, pt2, color, thickness, r, d):
            x1,y1 = pt1
            x2,y2 = pt2
        
            # Top left
            cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
            cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        
            # Top right
            cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
            cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        
            # Bottom left
            cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
            cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        
            # Bottom right
            cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
            cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

        # Create a numpy array from the lists above
        (images, labels) = [numpy.array(lists) for lists in [images, labels]]

        # OpenCV trains a model from the images using the Local Binary Patterns algorithm
        model = cv2.face.LBPHFaceRecognizer_create()
        # train the LBP algorithm on the images and labels we provided above
        model.train(images, labels)

        face_cascade = cv2.CascadeClassifier(haar_file)
        webcam = cv2.VideoCapture(0)
        print('Classifier trained!')
        print('Attempting to recognize faces...')
        while True:
            (_, im) = webcam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # detect faces using the haar_cacade file
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                # colour = bgr format
            # draw a rectangle around the face and resizing/ grayscaling it
            # uses the same method as in the create_data.py file
                # cv2.rectangle(im,(x,y),(x + w,y + h),(0, 255, 255),2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                # try to recognize the face(s) using the resized faces we made above
                prediction = model.predict(face_resize)
                rounded_rectangle(im, (x, y), (x + w, y + h), (0, 255, 255), 2, 15, 30)
                # if face is recognized, display the corresponding name
                if prediction[1] < 74:
                    cv2.putText(im,'%s' % (names[prediction[0]].strip()),(x + 5, (y + 25) + h), cv2.FONT_HERSHEY_PLAIN,1.5,(20,185,20), 2)
                    # print the confidence level with the person's name to standard output
                    confidence = (prediction[1]) if prediction[1] <= 100.0 else 100.0
                    print("predicted person: {}, confidence: {}%".format(names[prediction[0]].strip(), round((confidence / 74.5) * 100, 2)))
                    # cv2.putText(im,'SUCCESS',(x + 5, (y + 25) + h), cv2.FONT_HERSHEY_PLAIN,1.5,(65,65, 255), 2)
                    # time.sleep(1)
                    # break
                    # close the webcam and all open windows                    
                    webcam.release()
                    cv2.destroyAllWindows()
                # if face is unknown (if classifier is not trained on this face), show 'Unknown' text...
                else:
                    cv2.putText(im,'Unknown',(x + 5, (y + 25) + h), cv2.FONT_HERSHEY_PLAIN,1.5,(65,65, 255), 2)
                    print("predicted person: Unknown")

            # show window and set the window title
            cv2.imshow('OpenCV Face Recognition -  esc to close', im)
            key = cv2.waitKey(10)
            # esc to quit applet
            if key == 27:
                break


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
