import setuptools

VERSION = '0.0.1'
NAME = '3D-SeResNet'
DESCRIPTION = 'A Python package that comprises 2 phases which are semantic segmentation to extract an object from images and 3D object reconstruction.'
MAINTAINER = 'Ilsoo Seong'

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
        'inference_3d.py'
        ]
    packages=['Reco3D'],
)
