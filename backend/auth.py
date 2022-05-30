# import pyrebase
#
# firebaseConfig = {
#     'apiKey': "AIzaSyC39835NOkD5LiTcbHxc9hydxh-mNYjR94",
#     'authDomain': "tfg-juanllamazares.firebaseapp.com",
#     'databaseURL': "https://tfg-juanllamazares-default-rtdb.europe-west1.firebasedatabase.app",
#     'projectId': "tfg-juanllamazares",
#     'storageBucket': "tfg-juanllamazares.appspot.com",
#     'messagingSenderId': "104283811235",
#     'appId': "1:104283811235:web:055bf872b1d119ae92fe16",
#     'measurementId': "G-YGW4RFYRF9"
# }
# firebase = pyrebase.initialize_app(firebaseConfig)
# auth = firebase.auth()
#
#
# def sign_up(email, password):
#     try:
#         auth.create_user_with_email_and_password(email, password)
#     except:
#         print("ERROR signup")
#
#
# def login(email, password):
#     try:
#         user = auth.sign_in_with_email_and_password(email, password)
#     except:
#         print("ERROR in login")
#
