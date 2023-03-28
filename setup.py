import os

from setuptools import setup

BASEDIR = os.path.abspath(os.path.dirname(__file__))


def required(requirements_file):
    """ Read requirements file and remove comments and empty lines. """
    with open(os.path.join(BASEDIR, requirements_file), 'r') as f:
        requirements = f.read().splitlines()
        if 'MYCROFT_LOOSE_REQUIREMENTS' in os.environ:
            print('USING LOOSE REQUIREMENTS!')
            requirements = [r.replace('==', '>=').replace('~=', '>=') for r in requirements]
        return [pkg for pkg in requirements
                if pkg.strip() and not pkg.startswith("#")]


setup(
    name='ovos-classifiers',
    version='0.0.0a1',
    packages=['ovos_classifiers',
              'ovos_classifiers.datasets',
              'ovos_classifiers.heuristics',
              'ovos_classifiers.skovos',
              'ovos_classifiers.skovos.features',
              "ovos_classifiers.tasks",
              "ovos_classifiers.utils"],
    url='https://github.com/OpenVoiceOS/ovos-classifiers',
    license='apache-2.0',
    author='jarbasai',
    include_package_data=True,
    extras_require={
        "sklearn": ["scikit-learn"]
    },
    install_requires=required("requirements.txt"),
    author_email='jarbasai@mailfence.com'
)
