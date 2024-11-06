from setuptools import setup, find_packages

setup(
    name="code-helper",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    url="",
    license="",
    author="Andrew Brookins",
    author_email="andrew.b@prefect.io",
    description="",
    entry_points={
        "console_scripts": [
            "index_files=code_helper.create_embeddings_async:main",
        ],
    },
)
