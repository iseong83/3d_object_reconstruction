import boto3
import os

def s3_list_subfolders(s3_bucket, s3_path):
    '''Returns a list of folders unGder a path inside a s3 bucket.'''

    client = boto3.client('s3')
    result = client.list_objects(Bucket=s3_bucket,Delimiter='/', Prefix=s3_path)
    return [res.get('Prefix') for res in result.get('CommonPrefixes')]

def s3_download_folder(s3_bucket, s3_path='./data'):
    '''Downloads all contents inside a s3 path inside a bucket to a local folder. Does not work for root s3 path.'''
    client = boto3.client('s3')
    result = client.list_objects(Bucket=s3_bucket, Prefix=s3_path)
    for res in result.get('Contents'):
        #client.download_file(bucket, file.get('Key'), dest_pathname)
        dest_pathname = os.path.join(local_dest, res.get('Key'))
        if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
        client.download_file(s3_bucket, res.get('Key'), dest_pathname)
        print('Downloaded : {}'.format(res.get('Key')))

def download_from_s3_folder(s3_bucket):
    s3_download_folder(s3_bucket)
    #download_folder = os.path.splitext(os.path.basename(link))[0]
    #archive = download_folder + ".tgz"

    #os.system("tar -xvzf {0}".format(archive))
    #os.rename(download_folder, "data/{}".format(download_folder))
    # os.system("rm -f {0}".format(archive))


if __name__ == '__main__':
    s3_bucket_name = 'shapenetv1'
    download_from_s3_folder(s3_bucket_name)
