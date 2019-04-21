import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="intelligent_water_droplet",
    version="0.0.2",
    author="Manu S Pillai",
    author_email="manupillai308@gmail.com",
    description="A python implementation of the Intelligent Water Droplet, a Swarm based optimization algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/manupillai308/Intelligent-Water-Droplet",
    packages=setuptools.find_packages(),
    install_requires=["numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
