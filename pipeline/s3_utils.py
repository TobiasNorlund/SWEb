import boto3
import sys
import threading
import os
import s3fs
from pathlib import Path
from luigi.contrib.s3 import S3Target as OriginalS3Target, S3Client
from botocore.client import Config, BaseClient
from boto3.s3.transfer import TransferConfig

"""
When calling ListObjects on a prefix including folders that are non-existent, it returns NoSuchKey or 
InternalError when in fact it should return an empty list.

We implement utilities to patch this here, both for boto3 clients as well as for luigi s3 clients
"""


def _custom_list_objects_handler(parsed, model, **kwargs):
    """
    Custom handler to modify the list_objects response if a NoSuchKey error is detected.
    """
    if 'Error' in parsed and parsed['Error']['Code'] in ('NoSuchKey', 'InternalError'):

        # Modify the response to simulate an empty list of objects
        parsed.clear()
        parsed.update({
            'ResponseMetadata': {
                'HTTPStatusCode': 200,
                'HTTPHeaders': {},
                'RetryAttempts': 0
            },
            'IsTruncated': False,
            'Marker': '',
            'Contents': [],
            'Delimiter': '',
            'MaxKeys': 1000,
            'CommonPrefixes': []
        })

        # Modify the http_response to reflect a successful operation
        http_response = kwargs['http_response']
        http_response.status_code = 200
        http_response.reason = 'OK'


def _patch_client(client):
    event_system = client.meta.events
    event_system.register('after-call.s3.ListObjects', _custom_list_objects_handler)
    event_system.register('after-call.s3.ListObjectsV2', _custom_list_objects_handler)


_boto3_client = None
def get_boto3_s3_client() -> BaseClient:
    global _boto3_client
    if not _boto3_client:
        _boto3_client = boto3.client(
            service_name="s3",
        )
    _patch_client(_boto3_client)
    return _boto3_client


_luigi_client = None
def get_luigi_s3_client():
    global _luigi_client
    if not _luigi_client:
        # S3Client uses the boto3 resources api
        _luigi_client = S3Client()

    # Get the underlying client from the resource
    s3_client = _luigi_client.s3.meta.client
    _patch_client(s3_client)
    return _luigi_client


_s3fs_client = None
def get_s3fs_client():
    global _s3fs_client
    if not _s3fs_client:
        _s3fs_client = s3fs.S3FileSystem()

    # Get the underlying client
    s3_client = _s3fs_client.s3
    _patch_client(s3_client)
    return _s3fs_client


# Set the default block size to something large, to effectively disable multipart upload
s3fs.S3FileSystem.default_block_size = 1024 ** 4


def upload_directory_to_s3(directory_path, s3_bucket_name, s3_prefix, s3_client):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            s3_file_path = os.path.join(s3_prefix, os.path.relpath(file_path, directory_path))
            s3_client.upload_file(
                file_path, 
                s3_bucket_name, 
                s3_file_path,
                Callback=ProgressPercentage(str(file_path)),
                # setting this arbitrarily high to 50GB since it seems multipart uploads are not supported on Edgelab
                Config=TransferConfig(multipart_threshold=50 * (1024**3))
            )


class S3Target(OriginalS3Target):
    def __init__(self, *args, client=None, **kwargs):
        if client is None:
            client = get_luigi_s3_client()
        
        super().__init__(
            *args,
            client=client,
            **kwargs
        )


class ProgressPercentage(object):

    def __init__(self, filename):
        self._filename = filename
        self._size = int(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100

            if percentage % 10 == 0:
                sys.stdout.write(
                    f"\r{self._filename}  {self._seen_so_far}/{self._size}  ({percentage:.2f}%)"
                )
                sys.stdout.flush()