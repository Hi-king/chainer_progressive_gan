from setuptools import setup, find_packages


def requirements():
    list_requirements = []
    with open('requirements.txt') as f:
        for line in f:
            list_requirements.append(line.rstrip())
    return list_requirements


setup(
    name='chainer_progressive_gan',
    version='0.0.1',
    author='Hi_king',
    author_email='hikingko1@gmail.com',
    url='http://signico.hi-king.me/',
    description='',
    long_description='''
    ''',
    install_requires=requirements(),
    packages=find_packages()+["sample"],
    zip_safe=False,
    package_data={'sample': ["*"]},

    include_package_data=True,
)
