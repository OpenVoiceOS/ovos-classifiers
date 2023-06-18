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


UTTERANCE_ENTRY_POINT = (
    'ovos-utterance-normalizer=ovos_classifiers.opm.heuristics:UtteranceNormalizerPlugin',
    # ovos-classifiers models
    'ovos-utterance-coref-normalizer=ovos_classifiers.opm:CoreferenceNormalizerPlugin'
)
SOLVER_ENTRY_POINT = (
    # nltk data dependent
    'ovos-question-solver-wordnet=ovos_classifiers.opm.nltk:WordnetSolverPlugin'
)
SUMMARIZER_ENTRY_POINT = (
    'ovos-summarizer-solver-wordfreq=ovos_classifiers.opm.heuristics:HeuristicSummarizerPlugin'
)
QA_ENTRY_POINT = (
    'ovos-evidence-solver-bm25=ovos_classifiers.opm.heuristics:BM25SolverPlugin'
)
KW_ENTRY_POINT = (
    'ovos-keyword-extractor-heuristic=ovos_classifiers.opm.heuristics:HeuristicKeywordExtractorPlugin',
    # nltk data dependent (stopwords)
    'ovos-keyword-extractor-rake=ovos_classifiers.opm.nltk:RakeExtractorPlugin'
)
COREF_ENTRY = (
    "ovos-coref-solver-heuristic=ovos_classifiers.opm.heuristics:HeuristicCoreferenceSolverPlugin",
    # ovos-classifiers models
    "ovos-classifiers-coref-solver=ovos_classifiers.opm:OVOSCoreferenceSolverPlugin"
)
POSTAG_ENTRY = (
    "ovos-postag-plugin-regex=ovos_classifiers.opm.heuristics:RegexPostagPlugin",
    # nltk data dependent
    "ovos-postag-plugin-nltk=ovos_classifiers.opm.nltk:NltkPostagPlugin",
    # ovos-classifiers models
    "ovos-classifiers-postag-plugin=ovos_classifiers.opm:OVOSPostagPlugin"
)
LANG_DETECT_ENTRY = (
    # nltk data dependent
    "ovos-lang-detect-ngram-lm=ovos_classifiers.opm.nltk:LMLangDetectPlugin"
)
G2P_ENTRY_POINT = (
    'ovos-g2p-plugin-heuristic-arpa=ovos_classifiers.opm.heuristics:ARPAHeuristicPhonemizerPlugin'
)


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
              "ovos_classifiers.opm",
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
        'neon.plugin.text': UTTERANCE_ENTRY_POINT,
        'neon.plugin.solver': SOLVER_ENTRY_POINT,
        'neon.plugin.lang.detect': LANG_DETECT_ENTRY,
        'opm.solver.summarization': SUMMARIZER_ENTRY_POINT,
        "opm.solver.reading_comprehension": QA_ENTRY_POINT,
        "intentbox.keywords": KW_ENTRY_POINT,
        "intentbox.coreference": COREF_ENTRY,
        "intentbox.postag": POSTAG_ENTRY,
        "ovos.plugin.g2p": G2P_ENTRY_POINT
    }
)
