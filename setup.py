from setuptools import setup, find_packages

setup(
    name="graph_pe",
    version="0.1.0",
    description="Graph Pursuit Evasion - A multi-agent reinforcement learning environment",
    author="DwZhang01",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "gymnasium>=0.28.1",
        "networkx>=2.6.3",
        "pettingzoo>=1.22.0",
        "matplotlib>=3.5.0",
    ],
    python_requires=">=3.7",
)
