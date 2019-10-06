import io
import setuptools

VERSION = '0.0.1'
NAME = '3D-SeResNet'
DESCRIPTION = 'A Python package that comprises 2 phases which are semantic segmentation to extract an object from images and 3D object reconstruction.'
MAINTAINER = 'Ilsoo Seong'

def read(path, encoding='utf-8'):
    with io.open(path, encoding=encoding) as f:
        content = f.read()
    return content

def get_requirements(path='./requirements.txt'):
    content = read(path)
    requirements = [req for req in content.split("\n")
                    if req != '' and not req.startswith('#')]
    return requirements

setuptools.setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    license='MIT',
    author=MAINTAINER,
    install_requires=get_requirements(),
    scripts=[
        'demo.py',
        'run.py',
        'tests/inference_3d.py',
        'scripts/setup_and_preprocess.py',
        'scripts/re_process.py',
        ],
    packages=['Reco3D'],
)
