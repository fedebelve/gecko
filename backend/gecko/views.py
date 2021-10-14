from rest_framework import generics, permissions, views
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser, FileUploadParser, MultiPartParser
from django.http import JsonResponse
from django.db import IntegrityError
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated, AllowAny
#from backend.gecko import utils
from gecko.utils import clasify, fill
from gecko.serializers import AnalizeSerializer
from user_profile.models import Profile
from user_profile.serializers import UserSigninSerializer, UserProfileSerializer, UserSerializer
import gecko.preprocess as pre 
import os
import cv2
import numpy
from gecko.settings import BASE_DIR, RN_INCEPTION_MODEL, RN_VALIDATOR_MODEL
import tensorflow as tf
from tensorflow import keras 
from keras.applications. inception_v3 import InceptionV3
import base64
import PIL.Image as Image
from io import BytesIO 
from rest_framework.status import (HTTP_400_BAD_REQUEST, HTTP_200_OK, HTTP_201_CREATED, HTTP_401_UNAUTHORIZED, HTTP_404_NOT_FOUND)
from rest_framework.response import Response
from .authentication import token_expire_handler, expires_in
from rest_framework_api_key.permissions import HasAPIKey
from organization.permissions import HasOrganizationAPIKey
from django.contrib.auth.hashers import make_password
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Analize(views.APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser]

    def post(self, request, filename, format=None):
        img_path = f"{BASE_DIR}/tmp/{request.FILES['image']}"
        img = cv2.imdecode(numpy.fromstring(request.FILES['image'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(img_path, img,[int(cv2.IMWRITE_JPEG_QUALITY), 100])

        if self._validate(img_path):
            pre_processed_image = self._pre_process_image(img_path)
            response = self._process_image(pre_processed_image)

        else:
            response = "La imagen no es apta para ser procesada."
        
        os.remove(img_path)

        return JsonResponse({'response': str(response)}, status=200)

    def _validate(self,img_path): return pre.brightness_level(img_path) < 100

    def _pre_process_image(self, image_path):
        diameter = 299
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

                cv2.imwrite('/home/fede/gecko/test.jpeg', pre_processed_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                success += 1
                return pre_processed_image

        except AttributeError as e:
            print(e)
            print("Could not preprocess {}...".format(image))

        return success

    def _process_image(self, image):
        img = image.reshape(1, 299, 299, 3)
        #print(f"Image shape:{img.shape}")
        result = RN_INCEPTION_MODEL.predict(img)

        return result[0][0]

class AnalizeBase64(views.APIView):
    #authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated | HasOrganizationAPIKey]
    parser_classes = [JSONParser]

    def post(self, request, format=None):
        data = JSONParser().parse(request)
        results = []

        for item in data['worklist']:

            item['img_bytes'] = item['img_bytes'] + fill(len(item['img_bytes']))
            img_bytes = base64.b64decode(item['img_bytes'])

            img = Image.open(BytesIO(img_bytes))
            img_path = f"{BASE_DIR}/tmp/{item['img_name']}"
            if img.mode != "RGB":
                img = img.convert("RGB")

            img.save(img_path, "jpeg")

            brightness_level_ok, is_retinography = self._validate(img_path)
            result = 0

            item_result = {'img_name': item['img_name']}

            if brightness_level_ok and is_retinography:
                pre_processed_image = self._pre_process_image(img_path, 299)
                ben_color_image = pre.load_ben_color(pre_processed_image)
                result = self._process_image(ben_color_image)

                result, description = clasify(result)
                result_code="OK"              

            else:
                if not brightness_level_ok:
                    description="La imagen no posee la calidad suficiente"
                    result_code="poorQualityImage"

                if not is_retinography:
                    description="La imagen no es una retinografia"
                    result_code="invalidImage"

            item_result.update(result=result, description=description, result_code=result_code)
            results.append(item_result)
            os.remove(img_path)

        print(results)
        return JsonResponse({'response': results}, status=200)

    def _validate(self, img_path):
        is_retinography = self._is_retinography(img_path)
        brightness_level_ok = self._check_brightness_level(img_path)

        return brightness_level_ok, is_retinography

    def _is_retinography(self, img_path):
        pre_processed_image = self._pre_process_image(img_path, 224)
        img = pre_processed_image.reshape(1, 224, 224, 3)
        
        if RN_VALIDATOR_MODEL.predict(img) < 0.5:
            return True

    def _check_brightness_level(self,img_path):
        return (25 < pre.brightness_level(img_path) < 150)

    def _pre_process_image(self, image_path, diameter = 299):

        success = 0
        try:

            image = cv2.imread(os.path.abspath(image_path), -1)
            pre_processed_image = pre._resize_and_center_fundus(image, diameter=diameter)

            if pre_processed_image is None:
                print("Could not preprocess {}...".format(image))
            else:
                success += 1
                return pre_processed_image

        except AttributeError as e:
            print(e)
            print("Could not preprocess {}...".format(image))

        return success

    def _process_image(self, image):       
        img = cv2.resize(image, (299,299), 3)
        imgg = img.reshape(1, 299, 299, 3)
        result = RN_INCEPTION_MODEL.predict(imgg)

        return result[0][0]


@api_view(["POST"])
@permission_classes((AllowAny,))
def signin(request):
    signin_serializer = UserSigninSerializer(data = request.data)
    if not signin_serializer.is_valid():
        return Response(signin_serializer.errors, status=HTTP_400_BAD_REQUEST)
    
    user = authenticate(username=signin_serializer.data['username'], password=signin_serializer.data['password'])

    if not user:
        return Response({'detail': 'Invalid Credentials or activate account'}, status=HTTP_401_UNAUTHORIZED)

    try:
        token = Token.objects.get(user=user)
        token.delete()
    finally:
        token, _ = Token.objects.get_or_create(user=user)


    is_expired, token = token_expire_handler(token)
    user_serialized = UserSigninSerializer(user)

    return Response({'token': token.key,'first_name':  user.first_name,'last_name': user.last_name, 'expires_in': expires_in(token)}, status=HTTP_200_OK)


@api_view(['POST'])
@permission_classes((AllowAny,))
@authentication_classes([])
def signup(request):

    data = JSONParser().parse(request)
    user_serializer = UserSerializer(data = {'username': data['username'], 'password': make_password(data['password']), 'first_name': data['first_name'], 'last_name': data['last_name'], 'email':data['email']})
    user_profile_serializer = UserProfileSerializer(data = {'nro_doc': data['nro_doc'], 'country': data['country'], 'birth_date': data['birth_date'], 'job_type': data['job_type'], 'institution': data['institution']})

    user_valid = user_serializer.is_valid()
    profile_valid = user_profile_serializer.is_valid()

    if not user_valid or not profile_valid:
        return Response({'user': user_serializer.errors, 'profile': user_profile_serializer.errors}, status=HTTP_400_BAD_REQUEST)

    user = user_serializer.save()
    Profile.objects.create(user=user, nro_doc=data['nro_doc'], country=data['country'], birth_date=data['birth_date'], job_type=data['job_type'], institution=data['institution'])
    
    return Response({}, status=HTTP_201_CREATED)