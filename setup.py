"""
MangaGen - AI-Powered Manga Generation Framework
"""

from setuptools import setup, find_packages
import os

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read the README file for the long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="mangagen",
    version="0.1.0",
    description="AI-Powered Manga Generation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Siddhanth P Vashist",
    author_email="your.email@example.com",
    url="https://github.com/YOUR_USERNAME/MangaGen",
    license="MIT",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Artistic Software",
    ],
    entry_points={
        'console_scripts': [
            'mangagen=generate_manga_story:main',
            'mangagen-simple=simple_manga_generator:main',
            'mangagen-enhanced=enhanced_manga_generator:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.json'],
    },
    keywords="manga, ai, stable diffusion, llm, groq, comic, image generation",
)
