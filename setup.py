import setuptools
import os

# Get the absolute path of requirements.txt
req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

# Read requirements.txt safely
with open(req_path, "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

# Read README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="realtimestt-onnx",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A realtime speech-to-text library using ONNX models with advanced voice activity detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/realtimestt-onnx",
    packages=setuptools.find_packages(include=["realtimestt_onnx"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires='>=3.10',
    license='MIT',
    install_requires=requirements,
    keywords="real-time, audio, transcription, speech-to-text, voice-activity-detection, VAD, onnx, speech-recognition, voice-assistants, audio-processing",
    include_package_data=True,
)
