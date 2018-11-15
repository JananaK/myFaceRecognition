"""
CSE 196 Face Recognition Project
Author: Will Chen, Simon Fong

What this script should do:
1. Start running the camera.
2. Detect a face, display it, and get confirmation from user.
3. Send it for classification and fetch result.
4. Show result on face display.
"""

import time,cv2, base64, requests
from picamera import PiCamera
from picamera.array import PiRGBArray

people = ["bob", "rick", "sam", "david", "john", "sarah", "jane", "kristin", "will", "esther", "paul", "ryan", "joy", "jessica", "nick", "steve", "karina", "daniel", "janice"]

# Font that will be written on the image
FONT = cv2.FONT_HERSHEY_SIMPLEX

# TODO: Declare path to face cascade
CASCADE_PATH = '/home/pi/myFaceRecognition/project_2/haarcascade_frontalface_default.xml'
    
def request_from_server(img):
    """ 
    Sends image to server for classification.
    
    :param img: Image array to be classified.
    :returns: Returns a dictionary containing label and cofidence.
    """
    # URL or PUBLIC DNS to your server
    URL = "http://34.219.182.193:8080/"
    
    # File name so that it can be temporarily stored.
    temp_image_name = 'temp.jpg'
    
    # TODO: Save image with name stored in 'temp_image_name'
    cv2.imwrite(temp_image_name, img);

    # Reopen image and encode in base64
    image = open(temp_image_name, 'rb') #open binary file in read mode
    image_read = image.read()
    image_64_encode = base64.encodestring(image_read)
     
    # Defining a params dict for the parameters to be sent to the API
    payload = {'image':image_64_encode}
     
    # Sending post request and saving the response as response object
    response = requests.post(url = URL, json = payload)
     
    print 'We are getting the prediction...'

    # Get prediction from response
    prediction = response.json()

    print 'We have the prediction'

    return prediction


def main():
    # 1. Start running the camera.
    # TODO: Initialize face detector
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    
    # Initialize camera and update parameters
    camera = PiCamera()
    width = 224
    height = 224
    camera.rotation = 180
    camera.resolution = (width, height)
    rawCapture = PiRGBArray(camera, size=(width, height))

    # Warm up camera
    print 'Let me get ready ... 2 seconds ...'
    time.sleep(2)
    print 'Starting ...'

    # 2. Detect a face, display it, and get confirmation from user.
    for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
        
        # Get image array from frame
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        frame = frame.array
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('image', img)
        cv2.waitKey(1)

        # TODO: Use face detector to get faces.
        # Be sure to save the faces in a variable called 'faces'

    	faces = face_cascade.detectMultiScale(img, 1.3, 5)

    	for (x, y, w, h) in faces:
	    cv2.rectangle(faces,(x,y),(x+w,y+h),(255,255,255),2)
           
	    faces = faces[y:y+h, x:x+w]

	    print('==================================')
            print('Face detected!')
            cv2.imshow('Face Image for Classification', frame)
            
            # Keep showing image until a key is pressed
            cv2.waitKey()
            answer = input('Confirm image (1-yes / 0-no): ')
            print('==================================')

            if(answer == 1):
                print('Let\'s see who you are...')
                
                # TODO: Get label and confidence using request_from_server
                dictionary = request_from_server(faces)
                label = dictionary['label']  
	        confidence = dictionary['confidence']  
                print('New result found!')

                # TODO: Display label on face image
                # Save what you want to write on image to 'result_to_display'
                # [OPTIONAL]: At this point you only have a number to display, 
                # you could add some extra code to convert your number to a 
                # name

	        person = people[label]

                cv2.putText(frame, str("Hi " + person), (10, 30), FONT, 1, (0, 255, 0), 2)
                cv2.imshow('Face Image for Classification', frame)
                cv2.waitKey()
                break
        
            # Delete image in variable so we can get the next frame
        rawCapture.truncate(0)
        
        print('Waiting for image...')
        time.sleep(1)


# Runs main if this file is run directly
if(__name__ == '__main__'):
    main()

