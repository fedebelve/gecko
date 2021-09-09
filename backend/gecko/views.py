from rest_framework import generics, permissions, views
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser, FileUploadParser, MultiPartParser
from django.http import JsonResponse
from django.db import IntegrityError
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from rest_framework.decorators import api_view, authentication_classes
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from gecko.serializers import AnalizeSerializer
from user_profile.models import Profile
import gecko.preprocess as pre 
import os
import cv2
import numpy
from gecko.settings import BASE_DIR

@api_view(['POST'])
@authentication_classes([])
def signup(request):

    if request.method == 'POST':
        try:
            data = JSONParser().parse(request)
            user = User.objects.create_user(data['username'], password=data['password'], first_name=data['first_name'], last_name=data['last_name'], email=data['email'])
            user.save()
            Profile.objects.create(user=user, nro_doc=data['nro_doc'], country=data['country'], birth_date=data['birth_date'], job_type=data['job_type'], institution=data['institution'])
            token = Token.objects.create(user=user)
            return JsonResponse({'token':str(token)}, status=201)
        except IntegrityError:
            return JsonResponse({'error':'That username has already been taken. Please choose a new username'}, status=400)


@api_view(['POST'])
@authentication_classes([])
def login(request):

    if request.method == 'POST':
        data = JSONParser().parse(request)
        user = authenticate(request, username=data['username'], password=data['password'])
        if user is None:
            return JsonResponse({'error':'Could not login. Please check username and password'}, status=400)
        else:
            try:
                token = Token.objects.get(user=user)
            except:
                token = Token.objects.create(user=user)
            return JsonResponse({'token':str(token)}, status=200)


class Analize(views.APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser]

    def post(self, request, filename, format=None):
        print('AAA')
        print(request.data['eye'])
        print(request.FILES['image'])
        img_path = f"{BASE_DIR}/tmp/{request.FILES['image']}"
        img = cv2.imdecode(numpy.fromstring(request.FILES['image'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(img_path, img,[int(cv2.IMWRITE_JPEG_QUALITY), 100])

        if self._validate(img_path):
            pre_processed_image = self._pre_process_image(img_path)
            result = self._process_image(pre_processed_image)
            response = "OK"
        else:
            response= "La imagen no es apta para ser procesada."
        
        os.remove(img_path)
        print('BBB')

        return JsonResponse({'response': response}, status=200)


    def _validate(self,img_path):

        if 20 < pre.brightness_level(img_path) < 100:
            return True
        else:
            return False    


    def _pre_process_image(self, image_path):

        diameter=299 
        success = 0
        try:
            # Load the image and clone it for output.
            image = cv2.imread(os.path.abspath(image_path), -1)
#            image = cv2.imread(os.path.abspath(f"{BASE_DIR}/tmp/{filename}"), -1)

            pre_processed_image = pre._resize_and_center_fundus(image, diameter=diameter)

            if pre_processed_image is None:
                print("Could not preprocess {}...".format(image))
            else:
                  # Get the save path for the processed image.
                # image_filename = pre._get_filename(image_path)
                # image_jpeg_filename = "{0}.jpg".format(os.path.splitext(
                #                         os.path.basename(image_filename))[0])
                # output_path = os.path.join(save_path, image_jpeg_filename)

                cv2.imwrite('/home/fede/gecko/test.jpeg', pre_processed_image,[int(cv2.IMWRITE_JPEG_QUALITY), 100])

                success += 1
                return pre_processed_image

        except AttributeError as e:
            print(e)
            print("Could not preprocess {}...".format(image))

        return success


    def _process_image(self,image):
        pass



