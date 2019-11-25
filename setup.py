import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="sigpropy",
    version="0.1.0",
    author="Joseh P Vantassel",
    author_email="jvantassel@utexas.edu",
    description="Tools for digital signal processing.",
    long_description=long_description,
    long_description_content_type="text/markdown.-",
    url="https://github.com/jpvantassel/signal-processing",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Liscence :: OSI Approvied :: MIT Liscence",
        "Operating System :: OS Independent",
    ],
)
