#!/usr/bin/env python3
import os

from setuptools import setup

BASEDIR = os.path.abspath(os.path.dirname(__file__))


def get_version():
    """ Find the version of the package"""
    version = None
    version_file = os.path.join(BASEDIR, 'ovos_classifiers', 'version.py')
    major, minor, build, alpha = (None, None, None, None)
    with open(version_file) as f:
        for line in f:
            if 'VERSION_MAJOR' in line:
                major = line.split('=')[1].strip()
            elif 'VERSION_MINOR' in line:
                minor = line.split('=')[1].strip()
            elif 'VERSION_BUILD' in line:
                build = line.split('=')[1].strip()
            elif 'VERSION_ALPHA' in line:
                alpha = line.split('=')[1].strip()

            if ((major and minor and build and alpha) or
                    '# END_VERSION_BLOCK' in line):
                break
    version = f"{major}.{minor}.{build}"
    if alpha and int(alpha) > 0:
        version += f"a{alpha}"
    return version


def package_files(directory):
    paths = []
    for (path, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


def required(requirements_file):
    """ Read requirements file and remove comments and empty lines. """
    with open(os.path.join(BASEDIR, requirements_file), 'r') as f:
        requirements = f.read().splitlines()
        if 'MYCROFT_LOOSE_REQUIREMENTS' in os.environ:
            print('USING LOOSE REQUIREMENTS!')
            requirements = [r.replace('==', '>=').replace('~=', '>=') for r in requirements]
        return [pkg for pkg in requirements
                if pkg.strip() and not pkg.startswith("#")]

extra_files = package_files('ovos_classifiers/res')


PLUGIN_ENTRY_POINT = (
    'ovos-utterance-normalizer=ovos_classifiers.opm:UtteranceNormalizer',
    'ovos-utterance-coref-normalizer=ovos_classifiers.opm:CoreferenceNormalizer'
)
SOLVER_ENTRY_POINT = 'ovos-question-solver-wordnet=ovos_classifiers.opm:WordnetSolver'
SUMMARIZER_ENTRY_POINT = 'ovos-summarizer-solver-nltk=ovos_classifiers.opm:NltkSummarizer'


setup(
    name='ovos-classifiers',
    version=get_version(),
    author='jarbasai',
    author_email='jarbasai@mailfence.com',
    url='https://github.com/OpenVoiceOS/ovos-classifiers',
    license='apache-2.0',
    packages=['ovos_classifiers',
              'ovos_classifiers.datasets',
              'ovos_classifiers.heuristics',
              'ovos_classifiers.skovos',
              'ovos_classifiers.skovos.features',
              "ovos_classifiers.tasks",
              "ovos_classifiers.utils"],
    include_package_data=True,
    package_data={"": extra_files},
    extras_require={
        "sklearn": ["scikit-learn"]
    },
    install_requires=required("requirements.txt"),
    zip_safe=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    entry_points={
        'neon.plugin.text': PLUGIN_ENTRY_POINT,
        'neon.plugin.solver': SOLVER_ENTRY_POINT,
        'opm.solver.summarization': SUMMARIZER_ENTRY_POINT
    }
)
