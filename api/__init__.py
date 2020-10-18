import tensorflow as tf
import boto3
import os
from dotenv import load_dotenv

#AWS S3 setup
load_dotenv()
ACCESS_KEY_ID = os.getenv("ACCESS_KEY_ID")
SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")

def download_model():
    """
    Downloads 'gan' directory from S3 containing a TF saved_model format model.

    Notes:
        Making use of boto3 requires AWS credentials to be configured on host machine.
        These are configured in the .env file
    """
    print("Downloading model...")
    session = boto3.Session(
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY)
    s3_resource = session.resource('s3')
    bucket = s3_resource.Bucket('pokemon-dne')
    #Make local directories
    if not os.path.isdir('gan/'):
        os.mkdir('gan/')
    if not os.path.isdir('gan/variables/'):
        os.mkdir('gan/variables/')
    #Not sure if this is necessary
    if not os.path.isdir('gan/assets/'):
        os.mkdir('gan/assets/')
    #Download model file
    model = 'gan/saved_model.pb'
    bucket.download_file(model, model)
    #Crawl through variables directory and download each file
    variables_prefix = 'gan/variables/'
    variables = bucket.objects.filter(Prefix=variables_prefix)
    for variable in variables:
        bucket.download_file(variable.key, variable.key)

#Load model and make generator a global variable
download_model()
model = tf.keras.models.load_model('gan/')
generator, _ = model.layers